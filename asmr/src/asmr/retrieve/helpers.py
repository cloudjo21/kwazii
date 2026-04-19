"""DocumentRetriever — multi-field retrieval façade.

Responsibilities:
- Hold QueryRouter + field_configs state.
- Select which fields to query (all / target / per-modality).
- Delegate fetch+aggregate work to pipeline.py.
- Optionally apply a learned aggregation head (retrieve_with_aggregation_head).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

from asmr.retrieve import pipeline
from asmr.retrieve.query import Query
from asmr.retrieve.retrievers import QueryRouter

if TYPE_CHECKING:
    pass


def _derive_query_embedding(
    query: Union[str, Query],
    query_encoder: Any,
) -> np.ndarray:
    """Extract a float32 embedding vector from query via query_encoder."""
    import torch as _torch

    text = query if isinstance(query, str) else query.get_text()
    enc = query_encoder.encode([text])
    if isinstance(enc, _torch.Tensor):
        return enc[0].detach().cpu().float().numpy()
    return np.asarray(enc[0], dtype=np.float32)


def _merge_multimodal_results(
    text_ids: list[str],
    text_scores: np.ndarray,
    image_ids: list[str],
    image_scores: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Union text and image result sets, keeping both contributions."""
    if not text_ids:
        return image_ids, image_scores
    if not image_ids:
        return text_ids, text_scores
    # Both non-empty: concatenate along the doc dimension (last axis).
    # Callers receive the raw score matrices; downstream re-ranking handles merging.
    all_ids = list(dict.fromkeys(text_ids + image_ids))  # deduplicate, preserve order
    f_text = text_scores.shape[0] if text_scores.ndim >= 1 else 0
    f_image = image_scores.shape[0] if image_scores.ndim >= 1 else 0
    d = len(all_ids)
    id2col = {doc_id: i for i, doc_id in enumerate(all_ids)}

    merged = np.zeros((max(f_text, f_image), d), dtype=np.float32)
    for fi, doc_id in enumerate(text_ids):
        col = id2col[doc_id]
        if text_scores.ndim == 2:
            merged[:f_text, col] = np.maximum(merged[:f_text, col], text_scores[:, fi])
        elif text_scores.ndim == 1:
            merged[0, col] = max(merged[0, col], text_scores[fi])

    for fi, doc_id in enumerate(image_ids):
        col = id2col[doc_id]
        if image_scores.ndim == 2:
            merged[:f_image, col] = np.maximum(
                merged[:f_image, col], image_scores[:, fi]
            )
        elif image_scores.ndim == 1:
            merged[0, col] = max(merged[0, col], image_scores[fi])

    return all_ids, merged


class DocumentRetriever:
    """Multi-field retrieval façade.

    Decides *which* fields to query based on query content and optional
    field_configs. Delegates actual fetch+aggregate work to pipeline.py.
    """

    def __init__(
        self,
        fields_retriever: QueryRouter,
        field_configs: Optional[Dict] = None,
    ):
        self.fields_retriever = fields_retriever
        self.field_configs = field_configs or {}

    async def _retrieve_with_fields(
        self,
        query: Query,
        fields: list[str],
        k: int,
    ) -> tuple[list[str], np.ndarray]:
        return await pipeline.retrieve_and_aggregate(
            query, fields, self.fields_retriever, k=k
        )

    async def retrieve(self, query: Union[str, Query], k: int):
        """Retrieve across all registered fields."""
        q = pipeline.normalize_query(query)
        return await self._retrieve_with_fields(q, self.fields_retriever.fields, k)

    async def smart_retrieve(self, query: Query, k: int):
        """Retrieve across fields compatible with the query's content types."""
        if not self.field_configs:
            return await self.retrieve(query, k)
        target_fields = [
            config.name for config in query.get_target_fields(self.field_configs)
        ]
        return await self._retrieve_with_fields(query, target_fields, k)

    async def multimodal_retrieve(self, query: Query, k: int):
        """Retrieve by splitting a multimodal query into text and image sub-queries."""
        if not query.is_multimodal():
            return await self.smart_retrieve(query, k)

        text_ids, text_scores = await self.smart_retrieve(
            Query.from_text(query.get_text(), query.get_text_data_type()), k
        )
        image_ids, image_scores = await self.smart_retrieve(
            Query.from_image(query.get_image(), query.get_image_data_type()), k
        )
        return _merge_multimodal_results(text_ids, text_scores, image_ids, image_scores)

    async def retrieve_with_aggregation_head(
        self,
        query: Union[str, Query],
        k: int,
        *,
        field_lex_dense_pairs: Optional[list[tuple[str, str]]] = None,
        aggregation_head: Optional[Any] = None,
        query_emb: Optional[np.ndarray] = None,
        query_encoder: Optional[Any] = None,
        head_device: Optional[Any] = None,
    ) -> tuple[list[str], np.ndarray]:
        """Retrieve then optionally apply a learned aggregation head.

        Step (a): retrieve scores.
        Step (b): return raw scores when no head is provided.
        Step (c): apply head with query embedding.
        """
        # (a) Retrieve scores
        if field_lex_dense_pairs is not None:
            doc_ids, scores = await pipeline.retrieve_and_aggregate_hybrid(
                query,
                field_lex_dense_pairs,
                self.fields_retriever,
                k=k,
            )
        else:
            doc_ids, scores = await self.retrieve(query, k)

        # (b) No head — return raw aggregate scores
        if aggregation_head is None:
            return doc_ids, scores

        # (c) Apply aggregation head
        import torch

        from asmr.train.inference import apply_aggregation_head

        if query_emb is None:
            if query_encoder is None:
                raise ValueError("aggregation_head requires query_emb or query_encoder")
            query_emb = _derive_query_embedding(query, query_encoder)

        f_num = scores.shape[0]
        d_num = scores.shape[-1]
        field_mask = np.ones((f_num, d_num), dtype=bool)
        dev = head_device or torch.device("cpu")

        return apply_aggregation_head(
            doc_ids,
            scores.astype(np.float32),
            field_mask,
            query_emb.astype(np.float32),
            aggregation_head,
            device=dev,
        )
