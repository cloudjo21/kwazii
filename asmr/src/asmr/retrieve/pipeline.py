"""Retrieval pipeline — orchestration layer between QueryRouter and DocumentRetriever.

Responsibilities:
- Normalize Union[str, Query] to Query (single coercion point).
- Fetch per-field hits from QueryRouter, handling Ok/Err transparently.
- Compose fetch + aggregate into end-to-end async pipelines.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Union

import numpy as np

from asmr.retrieve.aggregate import (
    HybridFieldHits,
    aggregate_field_scores_async,
    aggregate_field_scores_hybrid_async,
)
from asmr.retrieve.query import Query
from asmr.retrieve.result import Err
from asmr.retrieve.retrievers import TOP_K_RETRIEVE, QueryRouter

if TYPE_CHECKING:
    pass


def normalize_query(query: Union[str, Query]) -> Query:
    """Coerce a raw str into a text-only Query. Single coercion point for the system."""
    if isinstance(query, str):
        return Query.from_text(query)
    return query


async def fetch_field_hits(
    router: QueryRouter,
    field: str,
    query: Query,
    k: int,
) -> list[tuple[str, float]]:
    """Fetch and normalize hits for one field.

    Returns an empty list when the retriever signals incompatibility (Err),
    so callers never need to handle the Result themselves.
    """
    k = min(k, TOP_K_RETRIEVE)
    result = await asyncio.to_thread(router.retrieve, field, query, k)
    if isinstance(result, Err):
        return []
    ranking = result.value
    return [(item.doc_id, float(item.score)) for item in ranking.items]


async def retrieve_and_aggregate(
    query: Union[str, Query],
    fields: list[str],
    router: QueryRouter,
    *,
    k: int = 100,
) -> tuple[list[str], np.ndarray]:
    """Normalize query, fetch all fields concurrently, return [F, D] score matrix."""
    q = normalize_query(query)
    per_field_hits = list(
        await asyncio.gather(*[fetch_field_hits(router, f, q, k) for f in fields])
    )
    return await aggregate_field_scores_async(per_field_hits, fields, k=k)


async def retrieve_and_aggregate_hybrid(
    query: Union[str, Query],
    field_lex_dense_pairs: list[tuple[str, str]],
    router: QueryRouter,
    *,
    k: int = 100,
) -> tuple[list[str], np.ndarray]:
    """Normalize query, fetch lex+dense pairs concurrently, return [F, 2, D] score matrix."""
    if not field_lex_dense_pairs:
        return [], np.zeros((0, 2, 0), dtype=np.float32)

    q = normalize_query(query)
    hybrid_hits: list[HybridFieldHits] = []
    for lex_key, dense_key in field_lex_dense_pairs:
        lex, dense = await asyncio.gather(
            fetch_field_hits(router, lex_key, q, k),
            fetch_field_hits(router, dense_key, q, k),
        )
        hybrid_hits.append(
            HybridFieldHits(field_name=lex_key, lex_hits=lex, dense_hits=dense)
        )

    return await aggregate_field_scores_hybrid_async(hybrid_hits, k=k)
