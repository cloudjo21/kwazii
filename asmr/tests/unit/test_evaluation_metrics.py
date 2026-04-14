"""Tests for src/evaluation/metrics.py."""

from evaluation.metrics import hit_at_k, mean_reciprocal_rank, recall_at_k


def test_hit_at_k() -> None:
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, {"b"}, 1) == 0.0
    assert hit_at_k(ranked, {"a"}, 1) == 1.0
    assert hit_at_k(ranked, {"c"}, 3) == 1.0


def test_recall_at_k() -> None:
    ranked = ["x", "a", "b", "c"]
    rel = {"a", "b", "z"}
    assert abs(recall_at_k(ranked, rel, 2) - (1 / 3)) < 1e-6


def test_mrr() -> None:
    ranked = ["x", "a", "b"]
    assert abs(mean_reciprocal_rank(ranked, {"a"}) - 0.5) < 1e-6
    assert mean_reciprocal_rank(ranked, {"z"}) == 0.0
