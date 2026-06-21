"""QueryRouter integration for STaRK-Prime disk-backed hybrid retrieval."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np

from asmr.datasets.stark_prime.loader import PRIME_FIELD_NAMES
from asmr.index.models import FieldBasedRanking, FieldBasedRankingItem
from asmr.retrieve.aggregate import HybridFieldHits, aggregate_field_scores_hybrid_async
from asmr.retrieve.pipeline import fetch_field_hits, retrieve_and_aggregate_hybrid
from asmr.retrieve.query import Query
from asmr.retrieve.result import Err, Ok, Result
from asmr.retrieve.retrievers import BaseFieldRetriever, QueryRouter

if TYPE_CHECKING:
    from asmr.evaluation.stark_prime_disk_index_v2 import (
        PrimeDiskIndexStore,
        ShortlistTiming,
    )


def _field_slug(field_name: str) -> str:
    return field_name.replace(" ", "_")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _hits_to_ranking(
    field: str,
    hits: list[tuple[str, float]],
    query_text: str,
) -> FieldBasedRanking:
    items = [
        FieldBasedRankingItem(doc_id=doc_id, score=score) for doc_id, score in hits
    ]
    return FieldBasedRanking(
        field_name=field,
        query=query_text,
        items=items,
        total_retrieved=len(items),
    )


class PrimeSparseFieldRetriever(BaseFieldRetriever):
    """BM25 sparse retriever for one STaRK-Prime field."""

    def __init__(self, store: PrimeDiskIndexStore, field: str) -> None:
        self._store = store
        self._field = field

    def can_handle(self, query: Query) -> bool:
        return query.has_text()

    def retrieve(self, query: Query, k: int) -> Result[FieldBasedRanking, str]:
        if not self.can_handle(query):
            return Err("PrimeSparseFieldRetriever requires text content")
        terms = _tokenize(query.get_text())
        sparse = self._store.field_cache.get_sparse(self._field)
        hits = sparse.search_topk(terms, k)
        return Ok(_hits_to_ranking(self._field, hits, query.get_text()))


class PrimeDenseFieldRetriever(BaseFieldRetriever):
    """Dense retriever bound to a precomputed query embedding."""

    def __init__(
        self,
        store: PrimeDiskIndexStore,
        field: str,
        query_emb: np.ndarray,
    ) -> None:
        self._store = store
        self._field = field
        self._query_emb = query_emb

    def can_handle(self, query: Query) -> bool:
        return True

    def retrieve(self, query: Query, k: int) -> Result[FieldBasedRanking, str]:
        hits = self._store._dense_topk_field(self._field, self._query_emb, k)
        text = query.get_text() if query.has_text() else ""
        dense_key = f"{_field_slug(self._field)}_dense"
        return Ok(_hits_to_ranking(dense_key, hits, text))


def build_prime_hybrid_router(
    store: PrimeDiskIndexStore,
    query_emb: np.ndarray,
) -> QueryRouter:
    """Build a QueryRouter with 22×2 sparse+dense retrievers for STaRK-Prime."""
    from asmr.evaluation.router import build_hybrid_router

    return build_hybrid_router(store, query_emb, PRIME_FIELD_NAMES)


def prime_field_lex_dense_pairs() -> list[tuple[str, str]]:
    """Return (sparse_key, dense_key) pairs for all Prime fields."""
    from asmr.evaluation.router import field_lex_dense_pairs

    return field_lex_dense_pairs(PRIME_FIELD_NAMES)


def attach_field_mask(
    store: PrimeDiskIndexStore,
    doc_ids: list[str],
    scores: np.ndarray,
) -> np.ndarray:
    """Build [F, D] bool mask for shortlist documents."""
    f_num = len(store.field_names)
    d_num = len(doc_ids)
    mask = np.zeros((f_num, d_num), dtype=bool)
    for di, doc_id in enumerate(doc_ids):
        col = store.id_to_col[doc_id]
        mask[:, di] = store.field_mask[:, col]
    return mask


def scores_with_field_mask(
    store: PrimeDiskIndexStore,
    doc_ids: list[str],
    scores: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Attach corpus field_mask to hybrid shortlist scores [F,2,D]."""
    mask = attach_field_mask(store, doc_ids, scores)
    return doc_ids, scores, mask


async def shortlist_prime_hybrid_router(
    store: PrimeDiskIndexStore,
    query_text: str,
    query_emb: np.ndarray,
    k: int,
    *,
    parallel: bool = False,
    timing: ShortlistTiming | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Hybrid shortlist via QueryRouter with field_mask."""
    t_total = time.perf_counter()
    router = build_prime_hybrid_router(store, query_emb)
    pairs = prime_field_lex_dense_pairs()
    q = Query.from_text(query_text)

    if not parallel:
        doc_ids, scores = await retrieve_and_aggregate_hybrid(
            q,
            pairs,
            router,
            k=k,
        )
        if timing is not None:
            timing.total_ms = (time.perf_counter() - t_total) * 1000.0
        return scores_with_field_mask(store, doc_ids, scores)

    async def _one_field(fname: str) -> HybridFieldHits:
        slug = _field_slug(fname)
        lex_key = f"{slug}_sparse"
        dense_key = f"{slug}_dense"
        lex_hits, dense_hits = await asyncio.gather(
            fetch_field_hits(router, lex_key, q, k),
            fetch_field_hits(router, dense_key, q, k),
        )
        return HybridFieldHits(
            field_name=fname,
            lex_hits=lex_hits,
            dense_hits=dense_hits,
        )

    t0 = time.perf_counter()
    hybrid_hits = await asyncio.gather(
        *[_one_field(fname) for fname in store.field_names]
    )
    doc_ids, scores = await aggregate_field_scores_hybrid_async(
        list(hybrid_hits),
        k=k,
    )
    if timing is not None:
        timing.total_ms = (time.perf_counter() - t_total) * 1000.0
        timing.sparse_ms = (time.perf_counter() - t0) * 1000.0
    return scores_with_field_mask(store, doc_ids, scores)


async def retrieve_and_aggregate_prime_hybrid(
    store: PrimeDiskIndexStore,
    query_text: str,
    query_emb: np.ndarray,
    *,
    k: int = 100,
    parallel: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Retrieve hybrid shortlist via QueryRouter."""
    doc_ids, scores, _mask = await shortlist_prime_hybrid_router(
        store,
        query_text,
        query_emb,
        k,
        parallel=parallel,
    )
    return doc_ids, scores


async def retrieve_and_aggregate_prime_hybrid_from_hits(
    store: PrimeDiskIndexStore,
    query_text: str,
    query_emb: np.ndarray,
    *,
    k: int = 100,
) -> tuple[list[str], np.ndarray]:
    """Build [F,2,D] via aggregate pipeline from per-field hits."""
    doc_ids, scores, _mask = await shortlist_prime_hybrid_router(
        store,
        query_text,
        query_emb,
        k,
        parallel=True,
    )
    return doc_ids, scores
