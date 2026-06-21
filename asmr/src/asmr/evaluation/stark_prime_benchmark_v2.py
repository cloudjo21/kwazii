"""STaRK-Prime MFARAll benchmark v2 (disk-backed indexes)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import optim

from asmr.evaluation.metrics import hit_at_k, mean_reciprocal_rank, recall_at_k
from asmr.evaluation.stark_prime_benchmark import (
    MFAR_PRIME_TEST,
    BenchmarkMetrics,
    BenchmarkReport,
)
from asmr.datasets.stark_prime.loader import (
    PRIME_FIELD_NAMES,
    PrimeQuery,
    load_prime_queries,
)
from asmr.datasets.stark_prime.torch_dataset import _example_from_shortlist
from asmr.evaluation.stark_prime_disk_index import (
    PrimeDiskIndexStore,
    build_prime_disk_index,
    is_index_built,
)
from asmr.train.aggregation import MFARFieldAdapter
from asmr.train.config import TrainConfig
from asmr.datasets.stark_prime.torch_dataset import (
    StarkRankingDataset,
    StarkRankingExample,
    collate_stark_batch,
)
from asmr.train.inference import apply_aggregation_head
from asmr.train.query_encoder import HfQueryEncoder
from asmr.train.trainer import AggregationTrainer

logger = logging.getLogger(__name__)

_LOG_EVERY = 100


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def train_mfar_adapter_v2(
    train_queries: list[PrimeQuery],
    store: PrimeDiskIndexStore,
    encoder: HfQueryEncoder,
    *,
    shortlist_k: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> MFARFieldAdapter:
    f_num = len(PRIME_FIELD_NAMES)
    model = MFARFieldAdapter(encoder.embedding_dim, f_num, 2).to(device)
    cfg = TrainConfig(query_dim=encoder.embedding_dim)
    trainer = AggregationTrainer(model, cfg)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    examples: list[StarkRankingExample] = []
    total = len(train_queries)
    t0 = time.time()
    for qi, q in enumerate(train_queries):
        q_emb = encoder.encode([q.query])[0].numpy()
        doc_ids, scores, mask = store.shortlist_hybrid(
            q.query,
            q_emb,
            shortlist_k,
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


def evaluate_ranked(
    queries: list[PrimeQuery],
    rank_fn,
    *,
    eval_k: int = 20,
) -> BenchmarkMetrics:
    t0 = time.time()
    hits, recalls, mrrs = [], [], []
    total = len(queries)
    for qi, q in enumerate(queries):
        ranked = rank_fn(q)
        rel = {str(a) for a in q.answer_ids}
        hits.append(hit_at_k(ranked, rel, 1))
        recalls.append(recall_at_k(ranked, rel, 20))
        mrrs.append(mean_reciprocal_rank(ranked, rel))
        if (qi + 1) % _LOG_EVERY == 0 or qi + 1 == total:
            logger.info(
                "eval %d/%d, RSS=%.1fMB",
                qi + 1,
                total,
                _rss_mb(),
            )
    n = max(len(hits), 1)
    return BenchmarkMetrics(
        hit_at_1=sum(hits) / n,
        recall_at_20=sum(recalls) / n,
        mrr=sum(mrrs) / n,
        num_queries=len(queries),
        elapsed_sec=time.time() - t0,
    )


def run_benchmark_v2(
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
) -> BenchmarkReport:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    encoder = HfQueryEncoder(model_name=encoder_name, device=str(device))
    if not is_index_built(index_dir) or rebuild_index:
        logger.info("Building disk index at %s", index_dir)
        build_prime_disk_index(
            data_root,
            index_dir,
            encoder,
            encoder_name=encoder_name,
            rebuild=rebuild_index,
        )

    store = PrimeDiskIndexStore(index_dir)
    logger.info(
        "Loaded disk index: %d docs, %d fields, RSS=%.1fMB",
        store.num_docs,
        len(store.field_names),
        _rss_mb(),
    )

    train_queries = load_prime_queries(data_root, "train")
    test_queries = load_prime_queries(data_root, "test")
    if max_train_queries > 0:
        train_queries = train_queries[:max_train_queries]
    if max_eval_queries > 0:
        test_queries = test_queries[:max_eval_queries]

    adapter = train_mfar_adapter_v2(
        train_queries,
        store,
        encoder,
        shortlist_k=shortlist_k,
        epochs=train_epochs,
        lr=1e-3,
        device=device,
    )

    def rank_mfar_all(q: PrimeQuery) -> list[str]:
        q_emb = encoder.encode([q.query])[0].numpy()
        doc_ids, scores, mask = store.shortlist_hybrid(
            q.query,
            q_emb,
            shortlist_k,
        )
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

    paper = MFAR_PRIME_TEST["MFARAll"]
    ratio = {
        "hit@1": metrics.hit_at_1 / paper["hit@1"] if paper["hit@1"] else 0.0,
        "recall@20": (
            metrics.recall_at_20 / paper["recall@20"] if paper["recall@20"] else 0.0
        ),
        "mrr": metrics.mrr / paper["mrr"] if paper["mrr"] else 0.0,
    }

    notes = [
        "v2 disk-backed indexes (memmap dense + CSC sparse top-k).",
        "Encoder is frozen contriever-msmarco (not STaRK-finetuned).",
        "Phase 1 head-only training; mFAR paper jointly fine-tunes encoder + G.",
        f"Shortlist k={shortlist_k} per field×scorer; eval reports top-{eval_k}.",
    ]
    if max_eval_queries > 0 or max_train_queries > 0:
        notes.append("Partial run: max_train_queries/max_eval_queries limits applied.")

    return BenchmarkReport(
        dataset="STaRK-Prime",
        protocol="MFARAll-v2",
        encoder=encoder_name,
        shortlist_k=shortlist_k,
        train_epochs=train_epochs,
        metrics=metrics,
        mfar_paper_test=MFAR_PRIME_TEST,
        ratio_vs_mfar_mfarall=ratio,
        notes=notes,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="STaRK-Prime MFARAll benchmark v2",
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
    args = parser.parse_args()

    rebuild = os.getenv("ASMR_INDEX_REBUILD", "0") == "1"
    report = run_benchmark_v2(
        args.data_root,
        args.index_dir,
        encoder_name=args.encoder,
        shortlist_k=args.shortlist_k,
        train_epochs=args.train_epochs,
        max_train_queries=args.max_train_queries,
        max_eval_queries=args.max_eval_queries,
        eval_k=args.eval_k,
        rebuild_index=rebuild,
    )

    payload = {
        "dataset": report.dataset,
        "protocol": report.protocol,
        "encoder": report.encoder,
        "shortlist_k": report.shortlist_k,
        "train_epochs": report.train_epochs,
        "metrics": asdict(report.metrics),
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
