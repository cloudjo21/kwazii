import numpy as np
import faiss
import os
from typing import List, Optional
from pathlib import Path

from fde.base import BaseFdeEncoder
from fde.config import PromptType
from asmr.index.config import FieldConfig
from asmr.index.models import FieldBasedRanking, FieldBasedRankingItem


class TextEncodingIndexer:
    """Text encoding indexer using MultiModalFdeEncoder with FAISS support"""

    def __init__(self, encoder: BaseFdeEncoder, config: FieldConfig):
        self.encoder = encoder
        self.config = config
        self.doc_ids: List[str] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: Optional[int] = None

        # Initialize or load FAISS index
        self._initialize_faiss_index()

    def _initialize_faiss_index(self):
        """Initialize FAISS index from config path or create new one"""
        if self.config.faiss_index_path and os.path.exists(
                self.config.faiss_index_path):
            self.load_index(self.config.faiss_index_path)
        else:
            # Will be initialized when first documents are added
            self.index = None

    def _create_faiss_index(self, dimension: int):
        """Create a new FAISS index with specified dimension"""
        self.dimension = dimension
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(dimension)

    def add_documents(
        self,
        doc_ids: List[str],
        texts: List[str],
        prompt_type: PromptType = PromptType.PASSAGE,
    ):
        """Add documents to the index (batch processing)"""
        if not doc_ids or not texts:
            return

        if len(doc_ids) != len(texts):
            raise ValueError("Number of doc_ids must match number of texts")

        # Encode texts using the FDE encoder (batch processing)
        embeddings = self.encoder.encode_text(texts, prompt_type)

        # Initialize index if needed
        if self.index is None:
            self._create_faiss_index(embeddings.shape[1])

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings)

        # Keep track of doc IDs
        self.doc_ids.extend(doc_ids)

    def add_document(self,
                     doc_id: str,
                     text: str,
                     prompt_type: PromptType = PromptType.PASSAGE):
        """Add a single document (wrapper for batch method)"""
        self.add_documents([doc_id], [text], prompt_type)

    def search(
            self,
            query: str,
            k: int = 10,
            prompt_type: PromptType = PromptType.QUERY) -> FieldBasedRanking:
        """Search for similar texts"""
        if self.index is None or self.index.ntotal == 0:
            return FieldBasedRanking(field_name=self.config.name,
                                     query=query,
                                     items=[],
                                     total_retrieved=0)

        # Encode query
        query_embedding = self.encoder.encode_text([query], prompt_type)

        # Normalize query embedding
        faiss.normalize_L2(query_embedding)

        # Search using FAISS
        k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding, k)

        # Convert to results format
        ranking_items = []
        for i, (similarity, idx) in enumerate(zip(similarities[0],
                                                  indices[0])):
            if idx != -1:  # Valid result
                ranking_items.append(
                    FieldBasedRankingItem(doc_id=self.doc_ids[idx],
                                          score=float(similarity)))

        return FieldBasedRanking(field_name=self.config.name,
                                 query=query,
                                 items=ranking_items,
                                 total_retrieved=len(ranking_items))

    def save_index(self, filepath: Optional[str] = None):
        """Save the FAISS index to disk"""
        if filepath is None:
            filepath = self.config.faiss_index_path

        if filepath is None:
            raise ValueError(
                "No filepath provided and no faiss_index_path in config")

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, filepath)

            # Save doc_ids separately
            doc_ids_path = filepath + ".doc_ids.npy"
            np.save(doc_ids_path, self.doc_ids)

    def load_index(self, filepath: str):
        """Load the FAISS index from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"FAISS index file not found: {filepath}")

        # Load FAISS index
        self.index = faiss.read_index(filepath)
        self.dimension = self.index.d

        # Load doc_ids
        doc_ids_path = filepath + ".doc_ids.npy"
        if os.path.exists(doc_ids_path):
            self.doc_ids = np.load(doc_ids_path, allow_pickle=True).tolist()
        else:
            # Fallback: generate sequential doc_ids
            self.doc_ids = [f"doc_{i}" for i in range(self.index.ntotal)]

    def get_stats(self) -> dict:
        """Get indexer statistics"""
        return {
            "total_documents": len(self.doc_ids),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "faiss_index_path": self.config.faiss_index_path,
        }
