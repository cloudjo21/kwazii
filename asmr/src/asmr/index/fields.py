import abc
import numpy as np
from typing import List, Optional, Union
from PIL import Image

from asmr.index import bm25
from asmr.index.config import FieldConfig, TokenizerType, RepresentationType
from asmr.index.image_encoder import ImageEncodingIndexer
from asmr.index.text_encoder import TextEncodingIndexer
from asmr.tokenize import helpers
from fde import base
from fde.config import PromptType


class BaseFieldIndex(abc.ABC):
    """Abstract base class for all field indices"""
    
    def __init__(self, config: FieldConfig):
        self.config = config
        self.field_name = config.name
    
    @abc.abstractmethod
    def add_document(self, doc_id: str, content: Union[str, Image.Image]) -> None:
        """Add a single document to the index"""
        pass
    
    @abc.abstractmethod
    def add_documents(self, doc_ids: List[str], contents: List[Union[str, Image.Image]]) -> None:
        """Add multiple documents to the index (batch processing)"""
        pass
    
    @abc.abstractmethod
    def search(self, query: Union[str, Image.Image], k: int = 10) -> List[tuple]:
        """Search the index and return top-k results"""
        pass


class SparseFieldIndex(BaseFieldIndex):
    """Abstract base class for sparse field indices (like BM25)"""
    
    def __init__(self, config: FieldConfig):
        if config.representation_type != RepresentationType.SPARSE:
            raise ValueError("SparseFieldIndex requires SPARSE representation_type")
        super().__init__(config)
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Setup tokenizer based on config"""
        if self.config.tokenizer_type == TokenizerType.MORPH:
            self.tokenizer = helpers.TokenizerWrapper.from_morph_tokenizer()
        elif self.config.tokenizer_type == TokenizerType.SPLIT:
            self.tokenizer = helpers.TokenizerWrapper.from_split_tokenizer()
        elif self.config.tokenizer_type == TokenizerType.HF_AUTO:
            self.tokenizer = helpers.TokenizerWrapper.from_auto_tokenizer(self.config.model_path)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.config.tokenizer_type}")


class DenseFieldIndex(BaseFieldIndex):
    """Abstract base class for dense field indices (vector embeddings)"""
    
    def __init__(self, config: FieldConfig):
        if config.representation_type != RepresentationType.DENSE:
            raise ValueError("DenseFieldIndex requires DENSE representation_type")
        super().__init__(config)
        self.vectors: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
    
    @abc.abstractmethod
    def _encode_content(self, content: Union[str, Image.Image]) -> np.ndarray:
        """Encode content to dense vector representation"""
        pass


class SparseTextFieldIndex(SparseFieldIndex):
    """Sparse text field index using BM25"""

    def __init__(self, config: FieldConfig, index: bm25.BM25Index):
        super().__init__(config)
        self.index = index

    def add_document(self, doc_id: str, content: str) -> None:
        """Add a single text document to the BM25 index"""
        # Implementation depends on BM25Index interface
        # For now, delegate to batch method
        self.add_documents([doc_id], [content])
    
    def add_documents(self, doc_ids: List[str], contents: List[str]) -> None:
        """Add multiple text documents to the BM25 index (batch processing)"""
        if len(doc_ids) != len(contents):
            raise ValueError("Number of doc_ids must match number of contents")
        
        # Tokenize all documents in batch
        tokenized_docs = []
        for content in contents:
            tokens = self.tokenizer.tokenize(content)
            tokenized_docs.append(tokens)
        
        # Add to BM25 index (implementation depends on BM25Index interface)
        for doc_id, tokens in zip(doc_ids, tokenized_docs):
            # This would need to be implemented based on the actual BM25Index interface
            pass
    
    def search(self, query: str, k: int = 10) -> List[tuple]:
        """Search using BM25 scoring"""
        terms = self.tokenizer.tokenize(query)
        scores = []
        for doc_id in range(self.index.shape[0]):
            doc_score = self.index.get_score(doc_id, terms)
            total_score = sum(ts.score for ts in doc_score.term_scores)
            if total_score > 0:
                scores.append((doc_id, total_score))
        # Sort by score descending and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class DenseTextFieldIndex(DenseFieldIndex):
    """Dense text field index using vector embeddings"""
    
    def __init__(self, config: FieldConfig, encoder: base.BaseFdeEncoder):
        super().__init__(config)
        self.indexer = TextEncodingIndexer(encoder, config)
    
    def _encode_content(self, content: str) -> np.ndarray:
        """Encode text content to dense vector"""
        return self.indexer.encoder.encode_text([content], PromptType.PASSAGE)[0]
    
    def add_document(self, doc_id: str, content: str) -> None:
        """Add a single text document to the dense index"""
        self.indexer.add_document(doc_id, content, PromptType.PASSAGE)
    
    def add_documents(self, doc_ids: List[str], contents: List[str]) -> None:
        """Add multiple text documents to the dense index (batch processing)"""
        self.indexer.add_documents(doc_ids, contents, PromptType.PASSAGE)
    
    def search(self, query: str, k: int = 10) -> List[tuple]:
        """Search using vector similarity"""
        return self.indexer.search(query, k, PromptType.QUERY)
    
    def save_index(self):
        """Save the index to the configured path"""
        self.indexer.save_index()
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        return self.indexer.get_stats()


class DenseImageFieldIndex(DenseFieldIndex):
    """Dense image field index using vector embeddings"""
    
    def __init__(self, config: FieldConfig, encoder: base.BaseFdeEncoder):
        super().__init__(config)
        self.indexer = ImageEncodingIndexer(encoder, config)
    
    def _encode_content(self, content: Image.Image) -> np.ndarray:
        """Encode image content to dense vector"""
        return self.indexer.encoder.encode_image([content])[0]
    
    def add_document(self, doc_id: str, content: Union[str, Image.Image]) -> None:
        """Add a single image document to the dense index"""
        self.indexer.add_document(doc_id, content)
    
    def add_documents(self, doc_ids: List[str], contents: List[Union[str, Image.Image]]) -> None:
        """Add multiple image documents to the dense index (batch processing)"""
        self.indexer.add_documents(doc_ids, contents)
    
    def search(self, query: Union[str, Image.Image], k: int = 10) -> List[tuple]:
        """Search using vector similarity"""
        if isinstance(query, str):
            # Text-to-image search
            return self.indexer.search_with_text(query, k)
        else:
            # Image-to-image search
            return self.indexer.search(query, k)
    
    def save_index(self):
        """Save the index to the configured path"""
        self.indexer.save_index()
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        return self.indexer.get_stats()


class DocumentIndex:
    def __init__(self):
        self.field_indices: dict[str, BaseFieldIndex] = dict()

    def add_field_index(self, field_name: str, index: BaseFieldIndex):
        self.field_indices[field_name] = index
