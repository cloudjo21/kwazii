"""STaRK evaluation runner stub — wire asmr index + qrels (§13.6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="STaRK-style eval (stub).")
    parser.add_argument(
        "--runs",
        type=Path,
        help="Path to run file (query_id -> ranked doc ids). JSON lines.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        help="Path to qrels (query_id -> set of relevant doc ids). JSON.",
    )
    args = parser.parse_args()

    if args.runs is None or args.qrels is None:
        print(
            "stark_eval: provide --runs and --qrels to score. "
            "See asmr.evaluation.metrics for Hit@1, R@20, MRR helpers."
        )
        return

    from asmr.evaluation.metrics import hit_at_k, mean_reciprocal_rank, recall_at_k

    qrels_raw = json.loads(args.qrels.read_text())
    qrels: dict[str, set[str]] = {k: set(v) for k, v in qrels_raw.items()}

    hits_1 = []
    recalls_20 = []
    mrrs = []
    for line in args.runs.read_text().splitlines():
        row = json.loads(line)
        qid = row["query_id"]
        ranked = [str(x) for x in row["doc_ids"]]
        rel = qrels.get(qid, set())
        hits_1.append(hit_at_k(ranked, rel, 1))
        recalls_20.append(recall_at_k(ranked, rel, 20))
        mrrs.append(mean_reciprocal_rank(ranked, rel))

    n = len(hits_1) or 1
    print(f"Hit@1: {sum(hits_1) / n:.4f}")
    print(f"Recall@20: {sum(recalls_20) / n:.4f}")
    print(f"MRR: {sum(mrrs) / n:.4f}")


if __name__ == "__main__":
    main()
