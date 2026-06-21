"""STaRK-MAG MFARAll benchmark runner (hybrid multi-field retrieval)."""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from asmr.datasets.stark_mag.loader import MagQuery, build_mag_corpus, load_mag_queries
from asmr.evaluation.metrics import hit_at_k, mean_reciprocal_rank, recall_at_k
from asmr.evaluation.stark_mag_indexes import MagFieldIndexes
from asmr.train.config import TrainMfarConfig
from asmr.train.inference import apply_aggregation_head
from asmr.train.query_encoder import HfQueryEncoder
from asmr.train.stark_mag_training import train_mfar_mag

logger = logging.getLogger(__name__)

# mFAR paper Table 1 — STaRK-MAG test-set numbers (MFARAll+2 best result)
MFAR_MAG_TEST = {
    "BM25": {"hit@1": 0.340, "recall@20": 0.609, "mrr": 0.430},
    "Contriever-FT": {"hit@1": 0.470, "recall@20": 0.695, "mrr": 0.556},
    "MFARAll": {"hit@1": 0.559, "recall@20": 0.741, "mrr": 0.643},
}


@dataclass
class BenchmarkMetrics:
    hit_at_1: float
    recall_at_20: float
    mrr: float
    num_queries: int
    elapsed_sec: float


@dataclass
class BenchmarkReport:
    dataset: str
    protocol: str
    encoder: str
    shortlist_k: int
    train_epochs: int
    metrics: BenchmarkMetrics
    mfar_paper_test: dict[str, dict[str, float]]
    ratio_vs_mfar_mfarall: dict[str, float]
    notes: list[str]


def evaluate_ranked(
    queries: list[MagQuery],
    rank_fn,
    *,
    eval_k: int = 20,
) -> BenchmarkMetrics:
    t0 = time.time()
    hits, recalls, mrrs = [], [], []
    for q in queries:
        ranked = rank_fn(q)
        rel = {str(a) for a in q.answer_ids}
        hits.append(hit_at_k(ranked, rel, 1))
        recalls.append(recall_at_k(ranked, rel, 20))
        mrrs.append(mean_reciprocal_rank(ranked, rel))
    n = max(len(hits), 1)
    return BenchmarkMetrics(
        hit_at_1=sum(hits) / n,
        recall_at_20=sum(recalls) / n,
        mrr=sum(mrrs) / n,
        num_queries=len(queries),
        elapsed_sec=time.time() - t0,
    )


def run_benchmark(
    data_root: Path,
    *,
    encoder_name: str = "facebook/contriever-msmarco",
    shortlist_k: int = 100,
    train_epochs: int = 5,
    max_docs: int = -1,
    max_train_queries: int = -1,
    max_eval_queries: int = -1,
    eval_k: int = 20,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> BenchmarkReport:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    corpus = build_mag_corpus(data_root, max_docs=max_docs)
    logger.info("MAG corpus: %d docs", len(corpus.doc_ids))

    encoder = HfQueryEncoder(model_name=encoder_name, device=str(device))
    indexes = MagFieldIndexes(corpus, encoder)

    train_queries = load_mag_queries(data_root, "train")
    test_queries = load_mag_queries(data_root, "test")
    if max_train_queries > 0:
        train_queries = train_queries[:max_train_queries]
    if max_eval_queries > 0:
        test_queries = test_queries[:max_eval_queries]

    cfg = TrainMfarConfig(
        shortlist_k=shortlist_k,
        epochs=train_epochs,
        adapter_lr=1e-3,
    )
    adapter = train_mfar_mag(
        train_queries,
        indexes,
        encoder,
        cfg,
        device,
        cache_dir=cache_dir,
        encoder_name=encoder_name,
    )

    def rank_mfar_all(q: MagQuery) -> list[str]:
        q_emb = encoder.encode([q.query])[0].numpy()
        doc_ids, scores, mask = indexes.shortlist_hybrid(q.query, q_emb, shortlist_k)
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
    indexes.close()

    paper = MFAR_MAG_TEST["MFARAll"]
    ratio = {
        "hit@1": metrics.hit_at_1 / paper["hit@1"] if paper["hit@1"] else 0.0,
        "recall@20": (
            metrics.recall_at_20 / paper["recall@20"] if paper["recall@20"] else 0.0
        ),
        "mrr": metrics.mrr / paper["mrr"] if paper["mrr"] else 0.0,
    }

    notes = [
        "Encoder is frozen contriever-msmarco (not STaRK-finetuned Contriever-FT).",
        "Phase 1 head-only training; mFAR paper jointly fine-tunes encoder + G.",
        f"Shortlist k={shortlist_k} per field×scorer; eval reports top-{eval_k}.",
        "MAG has 5 fields — abstract, author affiliation, cites, topic, title.",
    ]
    if max_docs > 0 or max_eval_queries > 0 or max_train_queries > 0:
        notes.append(
            "Partial run: max_docs/max_train_queries/max_eval_queries limits applied."
        )

    return BenchmarkReport(
        dataset="STaRK-MAG",
        protocol="MFARAll",
        encoder=encoder_name,
        shortlist_k=shortlist_k,
        train_epochs=train_epochs,
        metrics=metrics,
        mfar_paper_test=MFAR_MAG_TEST,
        ratio_vs_mfar_mfarall=ratio,
        notes=notes,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="STaRK-MAG MFARAll benchmark")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("asmr/data/stark_mag"),
    )
    parser.add_argument("--encoder", default="facebook/contriever-msmarco")
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=-1)
    parser.add_argument("--max-train-queries", type=int, default=-1)
    parser.add_argument("--max-eval-queries", type=int, default=-1)
    parser.add_argument("--eval-k", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_benchmark(
        args.data_root,
        encoder_name=args.encoder,
        shortlist_k=args.shortlist_k,
        train_epochs=args.train_epochs,
        max_docs=args.max_docs,
        max_train_queries=args.max_train_queries,
        max_eval_queries=args.max_eval_queries,
        eval_k=args.eval_k,
        cache_dir=args.cache_dir,
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
