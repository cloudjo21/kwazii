"""Tests for hybrid aggregate shortlist."""

import asyncio
from unittest.mock import Mock

import pytest

from asmr.index.models import FieldBasedRanking, FieldBasedRankingItem
from asmr.retrieve.aggregate import aggregate_field_scores_hybrid_async
from asmr.retrieve.query import Query
from asmr.retrieve.retrievers import QueryRouter, SparseTextFieldRetriever


def _ranking(field: str, pairs: list[tuple[str, float]]) -> FieldBasedRanking:
    items = [
        FieldBasedRankingItem(doc_id=d, score=s) for d, s in pairs
    ]
    return FieldBasedRanking(
        field_name=field,
        query="q",
        items=items,
        total_retrieved=len(items),
    )


def test_hybrid_two_fields_union_and_shape() -> None:
    """[F=2, M=2, D] with str doc ids."""

    async def _run() -> None:
        mock_sparse_a = Mock()
        mock_sparse_a.search.return_value = _ranking(
            "t_s", [("d1", 0.9), ("d2", 0.5)]
        )
        mock_dense_a = Mock()
        mock_dense_a.search.return_value = _ranking(
            "t_d", [("d2", 0.8), ("d3", 0.1)]
        )
        mock_sparse_b = Mock()
        mock_sparse_b.search.return_value = _ranking(
            "c_s", [("d1", 0.3)]
        )
        mock_dense_b = Mock()
        mock_dense_b.search.return_value = _ranking(
            "c_d", [("d3", 0.7)]
        )

        router = QueryRouter(
            {
                "t_s": SparseTextFieldRetriever(mock_sparse_a),
                "t_d": SparseTextFieldRetriever(mock_dense_a),
                "c_s": SparseTextFieldRetriever(mock_sparse_b),
                "c_d": SparseTextFieldRetriever(mock_dense_b),
            }
        )
        pairs = [("t_s", "t_d"), ("c_s", "c_d")]
        q = Query.from_text("hello")
        doc_ids, scores = await aggregate_field_scores_hybrid_async(
            q, pairs, router, k=10
        )
        assert doc_ids == ["d1", "d2", "d3"]
        assert scores.shape == (2, 2, 3)
        assert scores[0, 0, 0] == pytest.approx(0.9)
        assert scores[0, 1, 0] == pytest.approx(0.0)
        assert scores[0, 0, 1] == pytest.approx(0.5)
        assert scores[0, 1, 1] == pytest.approx(0.8)

    asyncio.run(_run())
