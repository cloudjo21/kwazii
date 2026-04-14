import asyncio
from typing import Tuple, Union

import numpy as np

from asmr.retrieve import retrievers
from asmr.retrieve.query import Query

# Lexical (m=0) and dense (m=1) slots for hybrid aggregation.
NUM_SCORER_KINDS = 2


def _normalize_doc_id(raw: object) -> str:
    """Unify document ids as str for dict keys (§13.1)."""
    if isinstance(raw, str):
        return raw
    return str(raw)


async def _fetch_field_hits(
    doc_retriever: retrievers.QueryRouter, field: str, query: Union[str, Query], k: int
):
    k = min(k, retrievers.TOP_K_RETRIEVE)

    # TODO support async retrieval in DocumentRetriever
    result = await asyncio.to_thread(doc_retriever.retrieve, field, query, k)
    hits = list(result)
    out = []
    for item in hits:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append((_normalize_doc_id(item[0]), float(item[1])))
        else:
            raise ValueError("doc_retriever.retrieve must yield (doc_id, score) pairs")
    return out


def _compute_cols_scores_sync(
    hits: list[tuple[str, float]], docid2col: dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    if not hits:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    ids_arr = np.array([h[0] for h in hits], dtype=object)
    sc_arr = np.fromiter((h[1] for h in hits), dtype=np.float32)
    cols = np.fromiter((docid2col.get(str(x), -1) for x in ids_arr), dtype=np.int64)
    valid = cols >= 0
    return cols[valid], sc_arr[valid]


async def _compute_cols_scores(
    hits: list[tuple[str, float]], docid2col: dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    return await asyncio.to_thread(_compute_cols_scores_sync, hits, docid2col)


async def aggregate_field_scores_async(
    query: Union[str, Query],
    fields: list[str],
    doc_retriever: retrievers.QueryRouter,
    k: int = 100,
) -> Tuple[list[str], np.ndarray]:
    # concurrently fetch top-k per field
    fetch_tasks = [
        _fetch_field_hits(doc_retriever, field, query, k) for field in fields
    ]
    per_field_hits = await asyncio.gather(*fetch_tasks)

    # union of doc ids (deterministic sorted order, str keys)
    union_set: set[str] = set()
    for hits in per_field_hits:
        for doc_id, _ in hits:
            union_set.add(_normalize_doc_id(doc_id))
    doc_ids = sorted(union_set)
    if not doc_ids:
        return [], np.zeros((len(fields), 0), dtype=np.float32)

    docid2col: dict[str, int] = {d: i for i, d in enumerate(doc_ids)}
    scores = np.zeros((len(fields), len(doc_ids)), dtype=np.float32)

    # compute cols and scores for each field in parallel (CPU work offloaded to threads)
    compute_tasks = [
        _compute_cols_scores(per_field_hits[i], docid2col) for i in range(len(fields))
    ]
    results = await asyncio.gather(*compute_tasks)

    # assign rows using numpy advanced indexing (done in the main thread)
    for fi, (cols, sc_arr) in enumerate(results):
        if cols.size:
            scores[fi, cols] = sc_arr

    return doc_ids, scores


async def aggregate_field_scores_hybrid_async(
    query: Union[str, Query],
    field_lex_dense_pairs: list[tuple[str, str]],
    doc_retriever: retrievers.QueryRouter,
    k: int = 100,
) -> Tuple[list[str], np.ndarray]:
    """Build [F, M, D] scores with M=2 (lexical, dense) per logical field.

    Each tuple is (router_key_lexical, router_key_dense) for the same logical
    field. Requires both keys to exist on ``doc_retriever``.

    Returns:
        doc_ids: str ids, length D
        scores: shape [F, 2, D] — [:,0,:] lexical, [:,1,:] dense
    """
    if not field_lex_dense_pairs:
        return [], np.zeros((0, NUM_SCORER_KINDS, 0), dtype=np.float32)

    f_num = len(field_lex_dense_pairs)
    fetch_tasks = []
    for lex_key, dense_key in field_lex_dense_pairs:
        fetch_tasks.append(_fetch_field_hits(doc_retriever, lex_key, query, k))
        fetch_tasks.append(_fetch_field_hits(doc_retriever, dense_key, query, k))
    all_hits = await asyncio.gather(*fetch_tasks)

    union_set: set[str] = set()
    for hits in all_hits:
        for doc_id, _ in hits:
            union_set.add(_normalize_doc_id(doc_id))
    doc_ids = sorted(union_set)
    if not doc_ids:
        return [], np.zeros((f_num, NUM_SCORER_KINDS, 0), dtype=np.float32)

    d_num = len(doc_ids)
    docid2col = {d: i for i, d in enumerate(doc_ids)}
    scores = np.zeros((f_num, NUM_SCORER_KINDS, d_num), dtype=np.float32)

    compute_tasks = []
    for fi in range(f_num):
        lex_hits = all_hits[fi * 2]
        dense_hits = all_hits[fi * 2 + 1]
        compute_tasks.append(_compute_cols_scores(lex_hits, docid2col))
        compute_tasks.append(_compute_cols_scores(dense_hits, docid2col))
    results = await asyncio.gather(*compute_tasks)

    for fi in range(f_num):
        cols_lex, sc_lex = results[fi * 2]
        cols_den, sc_den = results[fi * 2 + 1]
        if cols_lex.size:
            scores[fi, 0, cols_lex] = sc_lex
        if cols_den.size:
            scores[fi, 1, cols_den] = sc_den

    return doc_ids, scores


async def retrieve_documents(
    query: Union[str, Query],
    fields: list[str],
    doc_retriever: retrievers.QueryRouter,
    k: int = 100,
):
    return await aggregate_field_scores_async(query, fields, doc_retriever, k)


async def retrieve_documents_hybrid(
    query: Union[str, Query],
    field_lex_dense_pairs: list[tuple[str, str]],
    doc_retriever: retrievers.QueryRouter,
    k: int = 100,
):
    return await aggregate_field_scores_hybrid_async(
        query, field_lex_dense_pairs, doc_retriever, k
    )
