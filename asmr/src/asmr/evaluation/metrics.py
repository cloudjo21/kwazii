"""Hit@k, Recall@k, MRR for ranked doc-id lists (§13.6)."""

from __future__ import annotations


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant doc appears in top-k, else 0.0."""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    return 1.0 if relevant.intersection(top) else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """|relevant ∩ top-k| / |relevant| when relevant non-empty."""
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return len(relevant.intersection(top)) / len(relevant)


def mean_reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1/rank of first relevant doc, or 0.0 if none."""
    for idx, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / float(idx)
    return 0.0
