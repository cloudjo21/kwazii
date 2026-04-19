from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Lexical (m=0) and dense (m=1) slots for hybrid aggregation.
NUM_SCORER_KINDS = 2


def _normalize_doc_id(raw: object) -> str:
    """Unify document ids as str for dict keys (§13.1)."""
    if isinstance(raw, str):
        return raw
    return str(raw)


@dataclass(frozen=True)
class HybridFieldHits:
    """Named container for one logical field's lexical + dense hits.

    Replaces the fragile fi*2 / fi*2+1 positional indexing pattern.
    """

    field_name: str
    lex_hits: list[tuple[str, float]]
    dense_hits: list[tuple[str, float]]


def _compute_cols_scores(
    hits: list[tuple[str, float]], docid2col: dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    if not hits:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    ids_arr = np.array([h[0] for h in hits], dtype=object)
    sc_arr = np.fromiter((h[1] for h in hits), dtype=np.float32)
    cols = np.fromiter((docid2col.get(str(x), -1) for x in ids_arr), dtype=np.int64)
    valid = cols >= 0
    return cols[valid], sc_arr[valid]


async def aggregate_field_scores_async(
    per_field_hits: list[list[tuple[str, float]]],
    fields: list[str],
    *,
    k: int = 100,
) -> Tuple[list[str], np.ndarray]:
    """Aggregate pre-fetched per-field hits into a [F, D] score matrix.

    Args:
        per_field_hits: One list of (doc_id, score) pairs per field, same
            order as ``fields``.
        fields: Field names corresponding to each entry in per_field_hits.
        k: Maximum number of documents to retain (applied at caller level;
            kept here for interface consistency).

    Returns:
        doc_ids: Sorted union of all retrieved document ids (length D).
        scores: Float32 array of shape [F, D].
    """
    union_set: set[str] = set()
    for hits in per_field_hits:
        for doc_id, _ in hits:
            union_set.add(doc_id)  # already str — normalized by pipeline layer
    doc_ids = sorted(union_set)
    if not doc_ids:
        return [], np.zeros((len(fields), 0), dtype=np.float32)

    docid2col: dict[str, int] = {d: i for i, d in enumerate(doc_ids)}
    scores = np.zeros((len(fields), len(doc_ids)), dtype=np.float32)

    for fi, hits in enumerate(per_field_hits):
        cols, sc_arr = _compute_cols_scores(hits, docid2col)
        if cols.size:
            scores[fi, cols] = sc_arr

    return doc_ids, scores


async def aggregate_field_scores_hybrid_async(
    hybrid_hits: list[HybridFieldHits],
    *,
    k: int = 100,
) -> Tuple[list[str], np.ndarray]:
    """Build [F, 2, D] scores with M=2 (lexical=0, dense=1) per logical field.

    Args:
        hybrid_hits: One HybridFieldHits per logical field.
        k: Kept for interface consistency.

    Returns:
        doc_ids: Sorted union of all retrieved document ids (length D).
        scores: Float32 array of shape [F, 2, D].
    """
    if not hybrid_hits:
        return [], np.zeros((0, NUM_SCORER_KINDS, 0), dtype=np.float32)

    f_num = len(hybrid_hits)
    union_set: set[str] = set()
    for hf in hybrid_hits:
        for doc_id, _ in hf.lex_hits:
            union_set.add(doc_id)
        for doc_id, _ in hf.dense_hits:
            union_set.add(doc_id)

    doc_ids = sorted(union_set)
    if not doc_ids:
        return [], np.zeros((f_num, NUM_SCORER_KINDS, 0), dtype=np.float32)

    d_num = len(doc_ids)
    docid2col = {d: i for i, d in enumerate(doc_ids)}
    scores = np.zeros((f_num, NUM_SCORER_KINDS, d_num), dtype=np.float32)

    for fi, hf in enumerate(hybrid_hits):
        cols_lex, sc_lex = _compute_cols_scores(hf.lex_hits, docid2col)
        cols_den, sc_den = _compute_cols_scores(hf.dense_hits, docid2col)
        if cols_lex.size:
            scores[fi, 0, cols_lex] = sc_lex
        if cols_den.size:
            scores[fi, 1, cols_den] = sc_den

    return doc_ids, scores
