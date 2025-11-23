import abc
from typing import List, Union

from asmr.index import bm25
from asmr.index import fields
from asmr.retrieve.query import Query
from asmr.tokenize import helpers


TOP_K_RETRIEVE = 100


class BaseFieldRetriever(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, query: Union[str, Query], k: int) -> List[tuple]:
        pass


class SparseTextFieldRetriever(BaseFieldRetriever):
    """Retriever for sparse text fields (BM25)"""
    
    def __init__(self, field_index: fields.SparseTextFieldIndex):
        self.field_index = field_index
        self.tokenizer = field_index.tokenizer

    def retrieve(self, query: Union[str, Query], k: int) -> List[tuple]:
        if isinstance(query, Query):
            if not query.has_text():
                raise ValueError("SparseTextFieldRetriever requires text content in query")
            query_text = query.get_text()
        else:
            query_text = query
        
        return self.field_index.search(query_text, k)


class DenseTextFieldRetriever(BaseFieldRetriever):
    """Retriever for dense text fields"""
    
    def __init__(self, field_index: fields.DenseTextFieldIndex):
        self.field_index = field_index

    def retrieve(self, query: Union[str, Query], k: int) -> List[tuple]:
        if isinstance(query, Query):
            if not query.has_text():
                raise ValueError("DenseTextFieldRetriever requires text content in query")
            query_text = query.get_text()
        else:
            query_text = query
        
        return self.field_index.search(query_text, k)


class DenseImageFieldRetriever(BaseFieldRetriever):
    """Retriever for dense image fields"""
    
    def __init__(self, field_index: fields.DenseImageFieldIndex):
        self.field_index = field_index

    def retrieve(self, query: Union[str, Query], k: int) -> List[tuple]:
        if isinstance(query, Query):
            if query.has_text() and query.has_image():
                # For multimodal queries, we can use either text or image
                # Priority: use image if available, otherwise text
                query_content = query.get_image() if query.has_image() else query.get_text()
            elif query.has_text():
                # Text-to-image search
                query_content = query.get_text()
            elif query.has_image():
                # Image-to-image search
                query_content = query.get_image()
            else:
                raise ValueError("Query must have either text or image content")
        else:
            # Assume string query for text-to-image search
            query_content = query
        
        return self.field_index.search(query_content, k)


class MultiModalFieldRetriever(BaseFieldRetriever):
    """Retriever that can handle multimodal queries across different field types"""
    
    def __init__(self, text_field_index: fields.DenseTextFieldIndex = None, 
                 image_field_index: fields.DenseImageFieldIndex = None):
        self.text_field_index = text_field_index
        self.image_field_index = image_field_index
        
    def retrieve(self, query: Union[str, Query], k: int) -> List[tuple]:
        if isinstance(query, str):
            # String query - use text field if available
            if self.text_field_index is None:
                raise ValueError("No text field available for string query")
            return self.text_field_index.search(query, k)
        
        if not isinstance(query, Query):
            raise ValueError("Query must be string or Query object")
            
        results = []
        
        # Handle text content
        if query.has_text() and self.text_field_index:
            text_results = self.text_field_index.search(query.get_text(), k)
            results.extend(text_results)
        
        # Handle image content
        if query.has_image() and self.image_field_index:
            image_results = self.image_field_index.search(query.get_image(), k)
            results.extend(image_results)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


class QueryRouter:
    """Router to direct queries to appropriate field retrievers"""
    
    def __init__(self, field_retrievers: dict[str, BaseFieldRetriever]):
        self.field_retrievers = field_retrievers

    @property
    def fields(self) -> list[str]:
        return list(self.field_retrievers.keys())

    def retrieve(self, field: str, query: Union[str, Query], k: int) -> List[tuple]:
        """Retrieve using field name and query (supports both legacy string and Query object)"""
        if field not in self.field_retrievers:
            raise ValueError(f"Field '{field}' not found in QueryRouter.")
        retriever = self.field_retrievers[field]
        return retriever.retrieve(query, k)
    

    def smart_retrieve(self, query: Query, field_name: str, k: int) -> List[tuple]:
        """Smart retrieval for a single field that automatically determines best method based on query content"""
        if field_name not in self.field_retrievers:
            raise ValueError(f"Field '{field_name}' not found in QueryRouter.")
        retriever = self.field_retrievers[field_name]
        
        try:
            # Try retrieval based on field and query compatibility
            if isinstance(retriever, (SparseTextFieldRetriever, DenseTextFieldRetriever)):
                if query.has_text():
                    return retriever.retrieve(query, k)
            elif isinstance(retriever, DenseImageFieldRetriever):
                if query.has_image() or query.has_text():  # Support cross-modal
                    return retriever.retrieve(query, k)
            elif isinstance(retriever, MultiModalFieldRetriever):
                return retriever.retrieve(query, k)
            else:
                # Fallback: try direct retrieval
                return retriever.retrieve(query, k)
        except ValueError:
            # If incompatible, return empty list
            return []
    