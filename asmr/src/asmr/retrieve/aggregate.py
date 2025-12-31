import asyncio
from typing import Tuple, Union

import numpy as np

from asmr.retrieve import retrievers
from asmr.retrieve.query import Query


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
            out.append((int(item[0]), float(item[1])))
        else:
            raise ValueError("doc_retriever.retrieve must yield (doc_id, score) pairs")
    return out


def _compute_cols_scores_sync(
    hits, docid2col: dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    if not hits:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    ids_arr = np.fromiter((h[0] for h in hits), dtype=np.str_)
    sc_arr = np.fromiter((h[1] for h in hits), dtype=np.float32)
    cols = np.fromiter((docid2col.get(x, -1) for x in ids_arr), dtype=np.int64)
    valid = cols >= 0
    return cols[valid], sc_arr[valid]


async def _compute_cols_scores(
    hits, docid2col: dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    return await asyncio.to_thread(_compute_cols_scores_sync, hits, docid2col)


async def aggregate_field_scores_async(
    query: Union[str, Query],
    fields: list[str],
    doc_retriever: retrievers.QueryRouter,
    k: int = 100,
) -> Tuple[list[int], np.ndarray]:
    # concurrently fetch top-k per field
    fetch_tasks = [
        _fetch_field_hits(doc_retriever, field, query, k) for field in fields
    ]
    per_field_hits = await asyncio.gather(*fetch_tasks)

    # union of doc ids (deterministic sorted order)
    union_set = set()
    for hits in per_field_hits:
        for doc_id, _ in hits:
            union_set.add(doc_id)
    doc_ids = np.array(sorted(union_set), dtype=np.int64)
    if doc_ids.size == 0:
        return [], np.zeros((len(fields), 0), dtype=np.float32)

    docid2col: dict[str, int] = {d: i for i, d in enumerate(doc_ids)}
    scores = np.zeros((len(fields), doc_ids.size), dtype=np.float32)

    # compute cols and scores for each field in parallel (CPU work offloaded to threads)
    compute_tasks = [
        _compute_cols_scores(per_field_hits[i], docid2col) for i in range(len(fields))
    ]
    results = await asyncio.gather(*compute_tasks)

    # assign rows using numpy advanced indexing (done in main thread)
    for fi, (cols, sc_arr) in enumerate(results):
        if cols.size:
            scores[fi, cols] = sc_arr

    return doc_ids.tolist(), scores


async def retrieve_documents(
    query: Union[str, Query],
    fields: list[str],
    doc_retriever: retrievers.QueryRouter,
    k: int = 100,
):
    return await aggregate_field_scores_async(query, fields, doc_retriever, k)
