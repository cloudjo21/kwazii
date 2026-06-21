"""STaRK-Prime MFARAll benchmark v2.1 (cached indexes + FAISS + parallel)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import resource
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import optim

from asmr.evaluation.stark_prime_benchmark import (
    MFAR_PRIME_TEST,
    BenchmarkReport,
)
from asmr.datasets.stark_prime.loader import (
    PRIME_FIELD_NAMES,
    PrimeQuery,
    load_prime_queries,
)
from asmr.datasets.stark_prime.torch_dataset import _example_from_shortlist
from asmr.evaluation.stark_prime_benchmark_v2 import evaluate_ranked
from asmr.evaluation.stark_prime_disk_index import (
    PrimeDiskIndexStore,
    ShortlistTiming,
    build_prime_disk_index,
    is_index_built,
    migrate_index_v21,
)
from asmr.datasets.stark_prime.query_cache import (
    QueryEmbeddingCache,
    build_query_emb_cache,
    load_query_caches,
)
from asmr.train.aggregation import MFARFieldAdapter
from asmr.train.config import TrainConfig
from asmr.datasets.stark_prime.torch_dataset import StarkRankingDataset, collate_stark_batch
from asmr.train.inference import apply_aggregation_head
from asmr.train.query_encoder import HfQueryEncoder
from asmr.train.trainer import AggregationTrainer

logger = logging.getLogger(__name__)

_LOG_EVERY = 100


@dataclass
class LatencyStats:
    """Query latency percentiles in seconds."""

    p50: float = 0.0
    p95: float = 0.0
    mean: float = 0.0
    samples: list[float] = field(default_factory=list)


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _latency_stats(samples: list[float]) -> LatencyStats:
    if not samples:
        return LatencyStats()
    ordered = sorted(samples)
    n = len(ordered)
    p50 = ordered[n // 2]
    p95 = ordered[int(n * 0.95)] if n > 1 else ordered[0]
    return LatencyStats(
        p50=p50,
        p95=p95,
        mean=statistics.mean(samples),
        samples=samples,
    )


def _resolve_query_emb(
    query: PrimeQuery,
    encoder: HfQueryEncoder,
    caches: dict[str, QueryEmbeddingCache],
    split: str,
) -> np.ndarray:
    cache = caches.get(split)
    if cache is not None:
        return cache.get_or_encode(query, encoder)
    return encoder.encode([query.query])[0].numpy()


def train_mfar_adapter_v21(
    train_queries: list[PrimeQuery],
    store: PrimeDiskIndexStore,
    encoder: HfQueryEncoder,
    caches: dict[str, QueryEmbeddingCache],
    *,
    shortlist_k: int,
    epochs: int,
    lr: float,
    device: torch.device,
    parallel: bool = False,
    log_timing: bool = False,
) -> MFARFieldAdapter:
    f_num = len(PRIME_FIELD_NAMES)
    model = MFARFieldAdapter(encoder.embedding_dim, f_num, 2).to(device)
    cfg = TrainConfig(query_dim=encoder.embedding_dim)
    trainer = AggregationTrainer(model, cfg)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    examples: list = []
    total = len(train_queries)
    t0 = time.time()
    for qi, q in enumerate(train_queries):
        q_emb = _resolve_query_emb(q, encoder, caches, "train")
        doc_ids, scores, mask = store.shortlist_hybrid_dispatch(
            q.query,
            q_emb,
            shortlist_k,
            parallel=parallel,
            log_timing=log_timing and qi == 0,
        )
        if doc_ids:
            examples.append(
                _example_from_shortlist(q, doc_ids, scores, mask),
            )
        if (qi + 1) % _LOG_EVERY == 0 or qi + 1 == total:
            logger.info(
                "train shortlist %d/%d, RSS=%.1fMB, elapsed=%.0fs",
                qi + 1,
                total,
                _rss_mb(),
                time.time() - t0,
            )

    ds = StarkRankingDataset(examples)
    indices = list(range(len(ds)))
    logger.info(
        "Training MFARFieldAdapter on %d shortlists, RSS=%.1fMB",
        len(ds),
        _rss_mb(),
    )

    for epoch in range(1, epochs + 1):
        random.shuffle(indices)
        loss_sum = 0.0
        for idx in indices:
            batch = collate_stark_batch([ds[idx]], encoder, device)
            opt.zero_grad()
            out = trainer.training_step(batch)
            out.loss.backward()
            opt.step()
            loss_sum += out.loss.item()
        logger.info(
            "Epoch %d/%d loss=%.4f RSS=%.1fMB",
            epoch,
            epochs,
            loss_sum / max(len(indices), 1),
            _rss_mb(),
        )
    return model


def run_benchmark_v21(
    data_root: Path,
    index_dir: Path,
    *,
    encoder_name: str = "facebook/contriever-msmarco",
    shortlist_k: int = 100,
    train_epochs: int = 5,
    max_train_queries: int = -1,
    max_eval_queries: int = -1,
    eval_k: int = 20,
    seed: int = 42,
    rebuild_index: bool = False,
    migrate_faiss: bool = False,
    warm_query_cache: bool = False,
    parallel: bool | None = None,
) -> tuple[BenchmarkReport, LatencyStats]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if parallel is None:
        parallel = os.getenv("ASMR_V21_PARALLEL", "0") == "1"
    use_query_cache = warm_query_cache or os.getenv("ASMR_QUERY_CACHE", "0") == "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s, parallel=%s", device, parallel)

    encoder = HfQueryEncoder(model_name=encoder_name, device=str(device))
    if not is_index_built(index_dir) or rebuild_index:
        logger.info("Building disk index at %s", index_dir)
        build_faiss = migrate_faiss or os.getenv("ASMR_V21_MIGRATE_FAISS", "0") == "1"
        build_prime_disk_index(
            data_root,
            index_dir,
            encoder,
            encoder_name=encoder_name,
            rebuild=rebuild_index,
            build_faiss=build_faiss,
        )
    elif migrate_faiss or os.getenv("ASMR_V21_MIGRATE_FAISS", "0") == "1":
        logger.info("Migrating index to v2.1 FAISS at %s", index_dir)
        migrate_index_v21(index_dir)

    cache_dir = data_root / "cache"
    caches: dict[str, QueryEmbeddingCache] = {}
    if use_query_cache:
        if warm_query_cache:
            for split in ("train", "test"):
                build_query_emb_cache(
                    data_root,
                    split,
                    encoder,
                    cache_dir,
                )
        caches = load_query_caches(cache_dir)

    store = PrimeDiskIndexStore(index_dir)
    logger.info(
        "Loaded v2.1 index: backend=%s, %d docs, RSS=%.1fMB",
        store.dense_backend,
        store.num_docs,
        _rss_mb(),
    )

    train_queries = load_prime_queries(data_root, "train")
    test_queries = load_prime_queries(data_root, "test")
    if max_train_queries > 0:
        train_queries = train_queries[:max_train_queries]
    if max_eval_queries > 0:
        test_queries = test_queries[:max_eval_queries]

    adapter = train_mfar_adapter_v21(
        train_queries,
        store,
        encoder,
        caches,
        shortlist_k=shortlist_k,
        epochs=train_epochs,
        lr=1e-3,
        device=device,
        parallel=parallel,
    )

    latencies: list[float] = []

    def rank_mfar_all(q: PrimeQuery) -> list[str]:
        t0 = time.perf_counter()
        q_emb = _resolve_query_emb(q, encoder, caches, "test")
        timing = ShortlistTiming()
        doc_ids, scores, mask = store.shortlist_hybrid_dispatch(
            q.query,
            q_emb,
            shortlist_k,
            parallel=parallel,
            timing=timing,
        )
        latencies.append(time.perf_counter() - t0)
        if not doc_ids:
            return []
        sorted_ids, _ = apply_aggregation_head(
            doc_ids,
            scores,
            mask,
            q_emb,
            adapter,
            device=device,
        )
        return [str(x) for x in sorted_ids[:eval_k]]

    metrics = evaluate_ranked(test_queries, rank_mfar_all, eval_k=eval_k)
    latency = _latency_stats(latencies)
    logger.info(
        "Eval latency p50=%.3fs p95=%.3fs mean=%.3fs peak_rss=%.1fMB",
        latency.p50,
        latency.p95,
        latency.mean,
        _rss_mb(),
    )

    paper = MFAR_PRIME_TEST["MFARAll"]
    ratio = {
        "hit@1": metrics.hit_at_1 / paper["hit@1"] if paper["hit@1"] else 0.0,
        "recall@20": (
            metrics.recall_at_20 / paper["recall@20"] if paper["recall@20"] else 0.0
        ),
        "mrr": metrics.mrr / paper["mrr"] if paper["mrr"] else 0.0,
    }

    notes = [
        "v2.1: FieldIndexCache + optional FAISS dense + query emb cache.",
        f"dense_backend={store.dense_backend}, parallel={parallel}.",
        "Encoder is frozen contriever-msmarco (not STaRK-finetuned).",
        "Phase 1 head-only training; mFAR paper jointly fine-tunes encoder + G.",
        f"Shortlist k={shortlist_k} per field×scorer; eval reports top-{eval_k}.",
        f"Query latency p50={latency.p50:.3f}s, p95={latency.p95:.3f}s.",
    ]
    if max_eval_queries > 0 or max_train_queries > 0:
        notes.append("Partial run: max_train_queries/max_eval_queries limits applied.")

    report = BenchmarkReport(
        dataset="STaRK-Prime",
        protocol="MFARAll-v2.1",
        encoder=encoder_name,
        shortlist_k=shortlist_k,
        train_epochs=train_epochs,
        metrics=metrics,
        mfar_paper_test=MFAR_PRIME_TEST,
        ratio_vs_mfar_mfarall=ratio,
        notes=notes,
    )
    return report, latency


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="STaRK-Prime MFARAll benchmark v2.1",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/stark_prime"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/stark_prime/index/prime"),
    )
    parser.add_argument("--encoder", default="facebook/contriever-msmarco")
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--max-train-queries", type=int, default=-1)
    parser.add_argument("--max-eval-queries", type=int, default=-1)
    parser.add_argument("--eval-k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--warm-query-cache",
        action="store_true",
        help="Build query embedding cache before benchmark",
    )
    parser.add_argument(
        "--migrate-faiss",
        action="store_true",
        help="Migrate memmap dense to FAISS IndexFlatIP",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel per-field shortlist (22 fields)",
    )
    args = parser.parse_args()

    rebuild = os.getenv("ASMR_INDEX_REBUILD", "0") == "1"
    report, latency = run_benchmark_v21(
        args.data_root,
        args.index_dir,
        encoder_name=args.encoder,
        shortlist_k=args.shortlist_k,
        train_epochs=args.train_epochs,
        max_train_queries=args.max_train_queries,
        max_eval_queries=args.max_eval_queries,
        eval_k=args.eval_k,
        rebuild_index=rebuild,
        migrate_faiss=args.migrate_faiss,
        warm_query_cache=args.warm_query_cache,
        parallel=args.parallel if args.parallel else None,
    )

    payload = {
        "dataset": report.dataset,
        "protocol": report.protocol,
        "encoder": report.encoder,
        "shortlist_k": report.shortlist_k,
        "train_epochs": report.train_epochs,
        "metrics": asdict(report.metrics),
        "latency": {
            "p50_sec": latency.p50,
            "p95_sec": latency.p95,
            "mean_sec": latency.mean,
        },
        "mfar_paper_test": report.mfar_paper_test,
        "ratio_vs_mfar_mfarall": report.ratio_vs_mfar_mfarall,
        "notes": report.notes,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
