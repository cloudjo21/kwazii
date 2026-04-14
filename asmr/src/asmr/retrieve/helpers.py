from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

from asmr.retrieve import aggregate
from asmr.retrieve import retrievers
from asmr.retrieve.query import Query

if TYPE_CHECKING:
    import torch


class DocumentRetriever:
    def __init__(
        self,
        fields_retriever: retrievers.QueryRouter,
        field_configs: Optional[Dict] = None,
    ):
        self.fields_retriever = fields_retriever
        self.field_configs = field_configs or {}

    async def retrieve(self, query: Union[str, Query], k: int):
        """Traditional retrieval across all available fields."""
        return await aggregate.retrieve_documents(
            query, self.fields_retriever.fields, self.fields_retriever, k
        )

    async def smart_retrieve(self, query: Query, k: int):
        """Smart retrieval that uses Query's target field detection."""
        if not self.field_configs:
            return await self.retrieve(query, k)

        target_field_configs = query.get_target_fields(self.field_configs)
        target_field_names = [config.name for config in target_field_configs]
        return await aggregate.retrieve_documents(
            query, target_field_names, self.fields_retriever, k
        )

    async def multimodal_retrieve(self, query: Query, k: int):
        """Specialized retrieval for multimodal queries."""
        if not query.is_multimodal():
            return await self.smart_retrieve(query, k)

        text_results = []
        image_results = []

        if query.has_text():
            text_query = Query.from_text(query.get_text(), query.get_text_data_type())
            text_results = await self.smart_retrieve(text_query, k)

        if query.has_image():
            image_query = Query.from_image(
                query.get_image(), query.get_image_data_type()
            )
            image_results = await self.smart_retrieve(image_query, k)

        if text_results and image_results:
            return text_results
        if text_results:
            return text_results
        return image_results

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
        """Optional learned head on shortlist scores (§13.5).

        If ``aggregation_head`` is None, returns raw aggregate scores (numpy).

        ``query_emb`` shape [H], or pass ``query_encoder`` with ``.encode([text])``.
        """
        if field_lex_dense_pairs is not None:
            doc_ids, scores = await aggregate.retrieve_documents_hybrid(
                query,
                field_lex_dense_pairs,
                self.fields_retriever,
                k,
            )
        else:
            doc_ids, scores = await self.retrieve(query, k)

        if aggregation_head is None:
            return doc_ids, scores

        import torch

        from asmr.train.inference import apply_aggregation_head

        if query_emb is None:
            if query_encoder is None:
                raise ValueError(
                    "aggregation_head requires query_emb or query_encoder"
                )
            text = query if isinstance(query, str) else query.get_text()
            enc = query_encoder.encode([text])
            if isinstance(enc, torch.Tensor):
                query_emb = enc[0].detach().cpu().float().numpy()
            else:
                query_emb = np.asarray(enc[0], dtype=np.float32)

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
