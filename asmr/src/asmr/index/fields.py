import abc
import numpy as np
from typing import List, Optional, Union
from PIL import Image

from asmr.index import bm25
from asmr.index.bm25 import DocumentIndexToIdMapping
from asmr.index.config import FieldConfig, TokenizerType, RepresentationType
from asmr.index.models import FieldBasedRanking, FieldBasedRankingItem
from asmr.index.image_encoder import ImageEncodingIndexer
from asmr.index.text_encoder import TextEncodingIndexer
from asmr.tokenize import helpers
from asmr.tokenize.helpers import MorphTokenizerWrapper, TokenizerWrapper
from asmr import columnar
from asmr import vocab
from fde import base
from fde.config import PromptType


class BaseFieldIndex(abc.ABC):
    """Abstract base class for all field indices"""

    def __init__(self, config: FieldConfig):
        self.config = config
        self.field_name = config.name

    @abc.abstractmethod
    def add_document(self, doc_id: str, content: Union[str,
                                                       Image.Image]) -> None:
        """Add a single document to the index"""
        pass

    @abc.abstractmethod
    def add_documents(self, doc_ids: List[str],
                      contents: List[Union[str, Image.Image]]) -> None:
        """Add multiple documents to the index (batch processing)"""
        pass

    @abc.abstractmethod
    def search(self,
               query: Union[str, Image.Image],
               k: int = 10) -> FieldBasedRanking:
        """Search the index and return top-k results"""
        pass


class SparseFieldIndex(BaseFieldIndex):
    """Abstract base class for sparse field indices (like BM25)"""

    def __init__(self, config: FieldConfig):
        if config.representation_type != RepresentationType.SPARSE:
            raise ValueError(
                "SparseFieldIndex requires SPARSE representation_type")
        super().__init__(config)
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Setup tokenizer based on config"""
        if self.config.tokenizer_type == TokenizerType.MORPH:
            self.tokenizer = helpers.TokenizerWrapper.from_morph_tokenizer()
        elif self.config.tokenizer_type == TokenizerType.SPLIT:
            self.tokenizer = helpers.TokenizerWrapper.from_split_tokenizer()
        elif self.config.tokenizer_type == TokenizerType.HF_AUTO:
            self.tokenizer = helpers.TokenizerWrapper.from_auto_tokenizer(
                self.config.model_path)
        else:
            raise ValueError(
                f"Unsupported tokenizer type: {self.config.tokenizer_type}")


class DenseFieldIndex(BaseFieldIndex):
    """Abstract base class for dense field indices (vector embeddings)"""

    def __init__(self, config: FieldConfig):
        if config.representation_type != RepresentationType.DENSE:
            raise ValueError(
                "DenseFieldIndex requires DENSE representation_type")
        super().__init__(config)
        self.vectors: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []

    @abc.abstractmethod
    def _encode_content(self, content: Union[str, Image.Image]) -> np.ndarray:
        """Encode content to dense vector representation"""
        pass


class SparseTextFieldIndex(SparseFieldIndex):
    """Sparse text field index using BM25"""

    def __init__(self, config: FieldConfig, index: bm25.BM25Index = None):
        super().__init__(config)
        self.index = index
        self.doc_id_mapping: Optional[DocumentIndexToIdMapping] = None

    def add_document(self, doc_id: str, content: str) -> None:
        """Add a single text document to the BM25 index"""
        # Implementation depends on BM25Index interface
        # For now, delegate to batch method
        self.add_documents([doc_id], [content])

    def add_documents(self, doc_ids: List[str], contents: List[str]) -> None:
        """Add multiple text documents to the BM25 index (batch processing)"""
        if len(doc_ids) != len(contents):
            raise ValueError("Number of doc_ids must match number of contents")

        # Create tokenizer based on field config
        if self.config.tokenizer_type == TokenizerType.MORPH:
            tokenizer = TokenizerWrapper(MorphTokenizerWrapper())
        else:
            raise ValueError("Unsupported tokenizer type for BM25 indexing")

        # Collect all tokens from contents to build vocabulary
        all_tokens = set()
        for content in contents:
            tokens = tokenizer.tokenize(content)
            all_tokens.update(tokens)

        # Create vocabulary
        vocabulary = vocab.Vocabulary.from_token_set(all_tokens)

        # Create field-based columnar texts
        field_columnar_texts = columnar.FieldBasedColumnarTexts(
            [{
                self.field_name: content
            } for content in contents],
            self.field_name,
            len(contents),
        )

        # Build statistics
        field_stats = columnar.ColumnarStatisticsBuilder.build(
            field_columnar_texts, vocabulary)

        # Build BM25 index with doc_ids for mapping
        self.index = bm25.BM25Indexer.build(
            field_name=self.field_name,
            columnar_posting=contents,
            vocab=vocabulary,
            field_statistics=field_stats,
            doc_ids=doc_ids,
        )

        # Get doc_id_mapping from BM25Index
        self.doc_id_mapping = self.index.doc_id_mapping

    def search(self, query: str, k: int = 10) -> FieldBasedRanking:
        """Search using BM25 scoring"""
        terms = self.tokenizer.tokenize(query)
        scores = []
        print("#### Searching BM25 index...")
        print(f"#### index shape: {self.index.index.shape}")
        # TODO: search top-k ranking with terms, and have to return document_id of FieldIndex typed str in ranking
        for doc_pos in range(self.index.index.shape[0]):
            doc_score = self.index.get_score(doc_pos, terms)
            print(
                f"#### Doc Position: {doc_pos}, Term Scores: {[(ts.term, ts.score) for ts in doc_score.term_scores]}"
            )
            total_score = sum(ts.score for ts in doc_score.term_scores)
            if total_score > 0:
                # Convert int doc_id to str doc_id using mapping
                doc_id = self._get_doc_id(doc_pos)
                scores.append((doc_id, total_score))
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Create FieldBasedRanking items
        ranking_items = [
            FieldBasedRankingItem(doc_id=doc_id, score=score)
            for doc_id, score in scores[:k]
        ]

        return FieldBasedRanking(field_name=self.field_name,
                                 query=query,
                                 items=ranking_items,
                                 total_retrieved=len(scores))

    def _get_doc_id(self, doc_pos: int) -> str:
        """Convert integer doc_id to string doc_id using mapping"""
        if self.doc_id_mapping is None:
            return str(doc_pos)

        try:
            return self.doc_id_mapping.get_doc_id(doc_pos)
        except KeyError:
            return str(doc_pos)  # fallback


class DenseTextFieldIndex(DenseFieldIndex):
    """Dense text field index using vector embeddings"""

    def __init__(self, config: FieldConfig, encoder: base.BaseFdeEncoder):
        super().__init__(config)
        self.indexer = TextEncodingIndexer(encoder, config)

    def _encode_content(self, content: str) -> np.ndarray:
        """Encode text content to dense vector"""
        return self.indexer.encoder.encode_text([content],
                                                PromptType.PASSAGE)[0]

    def add_document(self, doc_id: str, content: str) -> None:
        """Add a single text document to the dense index"""
        self.indexer.add_document(doc_id, content, PromptType.PASSAGE)

    def add_documents(self, doc_ids: List[str], contents: List[str]) -> None:
        """Add multiple text documents to the dense index (batch processing)"""
        self.indexer.add_documents(doc_ids, contents, PromptType.PASSAGE)

    def search(self, query: str, k: int = 10) -> FieldBasedRanking:
        """Search using vector similarity"""
        results = self.indexer.search(query, k, PromptType.QUERY)

        # Convert tuple results to FieldBasedRanking
        ranking_items = [
            FieldBasedRankingItem(doc_id=doc_id, score=score)
            for doc_id, score in results
        ]

        return FieldBasedRanking(field_name=self.field_name,
                                 query=query,
                                 items=ranking_items,
                                 total_retrieved=len(results))

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

    def add_document(self, doc_id: str, content: Union[str,
                                                       Image.Image]) -> None:
        """Add a single image document to the dense index"""
        self.indexer.add_document(doc_id, content)

    def add_documents(self, doc_ids: List[str],
                      contents: List[Union[str, Image.Image]]) -> None:
        """Add multiple image documents to the dense index (batch processing)"""
        self.indexer.add_documents(doc_ids, contents)

    def search(self,
               query: Union[str, Image.Image],
               k: int = 10) -> FieldBasedRanking:
        """Search using vector similarity"""
        if isinstance(query, str):
            # Text-to-image search
            results = self.indexer.search_with_text(query, k)
            query_str = query
        else:
            # Image-to-image search
            results = self.indexer.search(query, k)
            query_str = "<image_query>"

        # Convert tuple results to FieldBasedRanking
        ranking_items = [
            FieldBasedRankingItem(doc_id=doc_id, score=score)
            for doc_id, score in results
        ]

        return FieldBasedRanking(field_name=self.field_name,
                                 query=query_str,
                                 items=ranking_items,
                                 total_retrieved=len(results))

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
