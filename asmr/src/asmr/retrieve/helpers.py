from typing import Union, Optional, Dict
from asmr.retrieve import aggregate
from asmr.retrieve import retrievers
from asmr.retrieve.query import Query


class DocumentRetriever:
    def __init__(
        self,
        fields_retriever: retrievers.QueryRouter,
        field_configs: Optional[Dict] = None,
    ):
        self.fields_retriever = fields_retriever
        self.field_configs = field_configs or {}

    async def retrieve(self, query: Union[str, Query], k: int):
        """Traditional retrieval across all available fields"""
        return await aggregate.retrieve_documents(
            query, self.fields_retriever.fields, self.fields_retriever, k
        )

    async def smart_retrieve(self, query: Query, k: int):
        """Smart retrieval that uses Query's target field detection"""
        if not self.field_configs:
            # Fallback to traditional retrieval
            return await self.retrieve(query, k)

        target_field_configs = query.get_target_fields(self.field_configs)
        target_field_names = [config.name for config in target_field_configs]
        return await aggregate.retrieve_documents(
            query, target_field_names, self.fields_retriever, k
        )

    async def multimodal_retrieve(self, query: Query, k: int):
        """Specialized retrieval for multimodal queries"""
        if not query.is_multimodal():
            return await self.smart_retrieve(query, k)

        # For multimodal queries, we might want to combine results differently
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

        # Combine and merge results (simple approach - could be made more sophisticated)
        if text_results and image_results:
            # Merge the document IDs and scores
            combined_doc_ids = list(set(text_results[0] + image_results[0]))
            # For now, return text results (could implement score fusion)
            return text_results
        elif text_results:
            return text_results
        else:
            return image_results
