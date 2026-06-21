"""Generic hybrid router builder — dataset-agnostic field×2 sparse+dense routing."""

import numpy as np
import numpy.typing as npt

from asmr.evaluation.stark_prime_router import (
    PrimeDenseFieldRetriever,
    PrimeSparseFieldRetriever,
)
from asmr.retrieve.retrievers import QueryRouter


def _field_slug(field_name: str) -> str:
    return field_name.replace(" ", "_")


def build_hybrid_router(
    store: object,
    query_emb: npt.NDArray[np.float32],
    field_names: tuple[str, ...],
) -> QueryRouter:
    """Build a QueryRouter with field×2 sparse+dense retrievers for any schema.

    Each field contributes one PrimeSparseFieldRetriever (BM25) and one
    PrimeDenseFieldRetriever (vector search).  The store must expose
    field_cache.get_sparse() and _dense_topk_field() (DiskIndexStore API).
    """
    from asmr.retrieve.retrievers import BaseFieldRetriever

    retrievers: dict[str, BaseFieldRetriever] = {}
    for fname in field_names:
        slug = _field_slug(fname)
        retrievers[f"{slug}_sparse"] = PrimeSparseFieldRetriever(store, fname)  # type: ignore[arg-type]
        retrievers[f"{slug}_dense"] = PrimeDenseFieldRetriever(store, fname, query_emb)  # type: ignore[arg-type]
    return QueryRouter(retrievers)


def field_lex_dense_pairs(
    field_names: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Return (sparse_key, dense_key) pairs for all fields in schema order."""
    return [
        (f"{_field_slug(fn)}_sparse", f"{_field_slug(fn)}_dense") for fn in field_names
    ]


__all__ = ["build_hybrid_router", "field_lex_dense_pairs"]
