"""STaRK-Prime MFARAll benchmark v3 (asmr modules + QueryRouter + JinaVera)."""

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

from asmr.evaluation.metrics import hit_at_k, mean_reciprocal_rank, recall_at_k
from asmr.evaluation.query_encoders import QueryEncoderProtocol, create_query_encoder
from asmr.evaluation.stark_prime_benchmark import (
    MFAR_PRIME_TEST,
    BenchmarkMetrics,
    BenchmarkReport,
)
from asmr.datasets.stark_prime.loader import (
    PrimeQuery,
    load_prime_queries,
)
from asmr.evaluation.stark_prime_disk_index_v2 import (
    PrimeDiskIndexStore,
    ShortlistTiming,
    build_prime_disk_index_v2,
    is_index_built,
    migrate_index_v21,
)
from asmr.datasets.stark_prime.query_cache import (
    QueryEmbeddingCache,
    build_query_emb_cache,
    load_query_caches,
    resolve_query_emb,
)
from asmr.datasets.stark_prime.shortlist_cache import (
    ShortlistCacheMissingError,
    spawn_build_shortlist_cache,
)
from asmr.train.stark_prime_training import (
    TrainMfarConfig,
    TrainingPhase,
    check_gate_90,
    train_mfar,
)
from asmr.train.inference import apply_aggregation_head

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
    encoder: QueryEncoderProtocol,
    caches: dict[str, QueryEmbeddingCache],
    split: str,
) -> np.ndarray:
    return resolve_query_emb(query, encoder, caches, split)


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


def run_benchmark_v3(
    data_root: Path,
    index_dir: Path,
    *,
    encoder_name: str = "jinavera",
    training_encoder_name: str | None = None,
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
    training_phase: int = 2,
    adapter_lr: float = 1e-3,
    encoder_lr: float = 2e-5,
    normalize_scores: bool = False,
    use_shortlist_cache: bool = False,
    rebuild_shortlist_cache: bool = False,
    require_shortlist_cache: bool = False,
    shortlist_chunk_size: int = 128,
    hf_model_name: str = "facebook/contriever-msmarco",
    lora_adapter_name: str = "stark_prime",
    lora_checkpoint: Path | str | None = None,
    truncate_dim: int = 1024,
    fde_output_dim: int | None = 1024,
) -> tuple[BenchmarkReport, LatencyStats]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if parallel is None:
        parallel = os.getenv("ASMR_V3_PARALLEL", "1") == "1"
    use_query_cache = warm_query_cache or os.getenv("ASMR_QUERY_CACHE", "0") == "1"
    use_shortlist_cache = (
        use_shortlist_cache
        or os.getenv(
            "ASMR_SHORTLIST_CACHE",
            "0",
        )
        == "1"
    )

    use_shortlist_cache = (
        use_shortlist_cache
        or os.getenv(
            "ASMR_SHORTLIST_CACHE",
            "0",
        )
        == "1"
    )
    require_shortlist_cache = (
        require_shortlist_cache
        or os.getenv(
            "ASMR_REQUIRE_SHORTLIST_CACHE",
            "0",
        )
        == "1"
    )
    if require_shortlist_cache:
        use_shortlist_cache = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Device: %s, parallel=%s, encoder=%s, truncate_dim=%d, fde_output_dim=%s",
        device,
        parallel,
        encoder_name,
        truncate_dim,
        fde_output_dim,
    )

    enc_kwargs = {
        "hf_model_name": hf_model_name,
        "device": str(device),
        "truncate_dim": truncate_dim,
        "fde_output_dim": fde_output_dim,
        "lora_adapter_name": lora_adapter_name,
        "lora_checkpoint": lora_checkpoint,
    }
    train_enc_key = training_encoder_name or encoder_name
    for_training = TrainingPhase(
        training_phase
    ) == TrainingPhase.JOINT and train_enc_key in (
        "jinavera-lora",
        "jina-lora",
        "jinavera",
        "jina",
    )
    encoder = create_query_encoder(
        train_enc_key if for_training else encoder_name,
        for_training=for_training,
        **enc_kwargs,
    )
    eval_encoder = encoder
    jina_lora_train = for_training and train_enc_key in (
        "jinavera-lora",
        "jina-lora",
    )
    if for_training and not jina_lora_train:
        eval_encoder = create_query_encoder(
            encoder_name,
            for_training=False,
            **enc_kwargs,
        )
    elif lora_checkpoint is not None and not jina_lora_train:
        eval_encoder = create_query_encoder(
            encoder_name,
            for_training=False,
            **enc_kwargs,
        )

    cache_dir = data_root / "cache"
    caches: dict[str, QueryEmbeddingCache] = {}
    if use_query_cache:
        if warm_query_cache:
            for split in ("train", "test"):
                build_query_emb_cache(
                    data_root,
                    split,
                    eval_encoder,
                    cache_dir,
                )
        caches = load_query_caches(cache_dir, encoder_name=eval_encoder.name)

    train_queries = load_prime_queries(data_root, "train")
    test_queries = load_prime_queries(data_root, "test")
    if max_train_queries > 0:
        train_queries = train_queries[:max_train_queries]
    if max_eval_queries > 0:
        test_queries = test_queries[:max_eval_queries]

    if rebuild_shortlist_cache:
        if not is_index_built(index_dir):
            msg = f"Cannot rebuild shortlist cache: index missing at {index_dir}"
            raise FileNotFoundError(msg)
        spawn_build_shortlist_cache(
            data_root=data_root,
            index_dir=index_dir,
            encoder_name=encoder_name,
            split="train",
            shortlist_k=shortlist_k,
            parallel=bool(parallel),
            chunk_size=shortlist_chunk_size,
            hf_model_name=hf_model_name,
            max_queries=max_train_queries,
        )

    defer_index_for_train = require_shortlist_cache and not rebuild_index
    store: PrimeDiskIndexStore | None = None

    if not defer_index_for_train:
        if not is_index_built(index_dir) or rebuild_index:
            logger.info("Building disk index at %s", index_dir)
            build_faiss = (
                migrate_faiss or os.getenv("ASMR_V3_MIGRATE_FAISS", "1") == "1"
            )
            build_prime_disk_index_v2(
                data_root,
                index_dir,
                eval_encoder,
                rebuild=rebuild_index,
                build_faiss=build_faiss,
            )
        elif migrate_faiss or os.getenv("ASMR_V3_MIGRATE_FAISS", "0") == "1":
            logger.info("Migrating index to FAISS at %s", index_dir)
            migrate_index_v21(index_dir)

        store = PrimeDiskIndexStore(index_dir)
        logger.info(
            "Loaded v3 index: backend=%s, %d docs, RSS=%.1fMB",
            store.dense_backend,
            store.num_docs,
            _rss_mb(),
        )

    adapter = train_mfar(
        train_queries,
        store,
        encoder,
        caches,
        TrainMfarConfig(
            shortlist_k=shortlist_k,
            epochs=train_epochs,
            adapter_lr=adapter_lr,
            encoder_lr=encoder_lr,
            phase=TrainingPhase(training_phase),
            normalize_scores=normalize_scores,
        ),
        device,
        parallel=parallel,
        cache_dir=cache_dir if use_shortlist_cache else None,
        rebuild_shortlist_cache=False,
        require_shortlist_cache=require_shortlist_cache,
        use_streaming_cache=True,
        chunk_size=shortlist_chunk_size,
    )

    if store is None:
        if not is_index_built(index_dir):
            msg = f"Index required for eval but missing at {index_dir}"
            raise FileNotFoundError(msg)
        store = PrimeDiskIndexStore(index_dir)
        logger.info(
            "Loaded v3 index for eval: backend=%s, %d docs, RSS=%.1fMB",
            store.dense_backend,
            store.num_docs,
            _rss_mb(),
        )

    latencies: list[float] = []

    def rank_mfar_all(q: PrimeQuery) -> list[str]:
        t0 = time.perf_counter()
        q_emb = _resolve_query_emb(q, eval_encoder, caches, "test")
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

    gate_pass, gate_checks = check_gate_90(ratio)
    phase_label = f"phase{training_phase}"

    notes = [
        "v3: QueryRouter + BM25Index.search_topk + PrimeDiskIndexStore v2.",
        f"encoder={eval_encoder.name}, dense_backend={store.dense_backend}.",
        f"training_encoder={encoder.name}, training={phase_label}, "
        f"adapter_lr={adapter_lr}, encoder_lr={encoder_lr}.",
        f"parallel={parallel}, normalize_scores={normalize_scores}.",
        f"shortlist_cache={use_shortlist_cache}, require_cache={require_shortlist_cache}.",
        f"Shortlist k={shortlist_k}; eval top-{eval_k}.",
        f"Query latency p50={latency.p50:.3f}s, p95={latency.p95:.3f}s.",
        f"90% gate: {'PASS' if gate_pass else 'FAIL'} {gate_checks}.",
    ]
    if max_eval_queries > 0 or max_train_queries > 0:
        notes.append("Partial run: max_train_queries/max_eval_queries limits applied.")

    report = BenchmarkReport(
        dataset="STaRK-Prime",
        protocol=f"MFARAll-v3-{phase_label}",
        encoder=eval_encoder.name,
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
        description="STaRK-Prime MFARAll benchmark v3",
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
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=1024,
        help="Jina MRL dim for LoRA training (128/256/512/1024/2048)",
    )
    parser.add_argument(
        "--fde-output-dim",
        type=int,
        default=1024,
        help="Jinavera FDE final_projection_dimension (default: same as truncate-dim)",
    )
    parser.add_argument(
        "--legacy-fde-prefix",
        action="store_true",
        help="Use full 10240-d FDE + prefix truncate (deprecated; ADR-003)",
    )
    parser.add_argument("--encoder", default="jinavera")
    parser.add_argument(
        "--training-encoder",
        default=None,
        help="Phase 2 training encoder (default: jinavera-lora when --encoder jinavera)",
    )
    parser.add_argument(
        "--lora-adapter-name",
        default="stark_prime",
        help="PEFT adapter slug for Jina LoRA training",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=None,
        help="LoRA weights path for eval live re-encode",
    )
    parser.add_argument(
        "--hf-model-name",
        default="facebook/contriever-msmarco",
        help="HF model id when --encoder contriever (STaRK-FT checkpoint path ok)",
    )
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument(
        "--training-phase",
        type=int,
        choices=[1, 2],
        default=2,
        help="1=head-only G; 2=encoder+G joint (Contriever or Jina LoRA)",
    )
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument(
        "--normalize-scores",
        action="store_true",
        help="Per-(field,scorer) batch normalization before head",
    )
    parser.add_argument(
        "--use-shortlist-cache",
        action="store_true",
        help="Load/build train shortlist disk cache",
    )
    parser.add_argument(
        "--rebuild-shortlist-cache",
        action="store_true",
        help="Spawn subprocess to rebuild train shortlist streaming cache",
    )
    parser.add_argument(
        "--require-shortlist-cache",
        action="store_true",
        help="Fail if train shortlist cache is missing; skip inline shortlist",
    )
    parser.add_argument(
        "--shortlist-chunk-size",
        type=int,
        default=128,
        help="Chunk size for v2 streaming shortlist cache",
    )
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
        help="Use parallel per-field shortlist via QueryRouter",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel shortlist",
    )
    args = parser.parse_args()

    parallel: bool | None = None
    if args.parallel:
        parallel = True
    if args.no_parallel:
        parallel = False

    training_encoder = args.training_encoder
    if training_encoder is None and args.encoder in ("jinavera", "jina"):
        training_encoder = "jinavera-lora"

    fde_output_dim: int | None = args.fde_output_dim
    if args.legacy_fde_prefix:
        fde_output_dim = None

    rebuild = os.getenv("ASMR_INDEX_REBUILD", "0") == "1"
    try:
        report, latency = run_benchmark_v3(
            args.data_root,
            args.index_dir,
            encoder_name=args.encoder,
            training_encoder_name=training_encoder,
            shortlist_k=args.shortlist_k,
            train_epochs=args.train_epochs,
            max_train_queries=args.max_train_queries,
            max_eval_queries=args.max_eval_queries,
            eval_k=args.eval_k,
            rebuild_index=rebuild,
            migrate_faiss=args.migrate_faiss,
            warm_query_cache=args.warm_query_cache,
            parallel=parallel,
            training_phase=args.training_phase,
            adapter_lr=args.adapter_lr,
            encoder_lr=args.encoder_lr,
            normalize_scores=args.normalize_scores,
            use_shortlist_cache=args.use_shortlist_cache,
            rebuild_shortlist_cache=args.rebuild_shortlist_cache,
            require_shortlist_cache=args.require_shortlist_cache,
            shortlist_chunk_size=args.shortlist_chunk_size,
            hf_model_name=args.hf_model_name,
            lora_adapter_name=args.lora_adapter_name,
            lora_checkpoint=args.lora_checkpoint,
            truncate_dim=args.truncate_dim,
            fde_output_dim=fde_output_dim,
        )
    except ShortlistCacheMissingError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    gate_pass, gate_checks = check_gate_90(report.ratio_vs_mfar_mfarall)
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
        "gate_90_pass": gate_pass,
        "gate_90_checks": gate_checks,
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
