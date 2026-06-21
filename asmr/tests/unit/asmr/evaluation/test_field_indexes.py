"""Unit tests for SparseTextFieldIndex and DenseTextFieldIndex with IndexUsage."""

import numpy as np
import pytest
from pathlib import Path

from fde.config import PromptType

from asmr.index.config import FieldConfig, IndexUsage, RepresentationType, TokenizerType
from asmr.index.fields import DenseTextFieldIndex, SparseTextFieldIndex


_DIM = 32


def _make_embedding(text: str, dim: int) -> np.ndarray:
    """Toy embedding: maps first char to a one-hot direction."""
    emb = np.zeros(dim, dtype=np.float32)
    emb[ord(text[0]) % dim] = 1.0
    return emb


class _StubEncoder:
    """Minimal TextEncoderProtocol implementation for testing."""

    @property
    def embedding_dim(self) -> int:
        return _DIM

    def encode_text(
        self,
        texts: list[str],
        prompt_type: PromptType,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        return np.stack([_make_embedding(t, _DIM) for t in texts])


_DOCS = ["apple pie recipe", "banana smoothie guide", "cherry jam tutorial"]
_DOC_IDS = ["doc_0", "doc_1", "doc_2"]


class TestFieldConfigIndexUsage:
    def test_dense_serving_without_faiss_path_raises(self) -> None:
        with pytest.raises(ValueError, match="faiss_index_path"):
            FieldConfig("f", TokenizerType.SPLIT, RepresentationType.DENSE)

    def test_dense_benchmark_without_faiss_path_ok(self) -> None:
        cfg = FieldConfig(
            "f",
            TokenizerType.SPLIT,
            RepresentationType.DENSE,
            usage=IndexUsage.BENCHMARK,
        )
        assert cfg.usage == IndexUsage.BENCHMARK

    def test_sparse_serving_default_ok(self) -> None:
        cfg = FieldConfig("f", TokenizerType.SPLIT, RepresentationType.SPARSE)
        assert cfg.usage == IndexUsage.SERVING


class TestSparseTextFieldIndex:
    def test_add_documents_and_search(self, tmp_path: Path) -> None:
        cfg = FieldConfig("content", TokenizerType.SPLIT, RepresentationType.SPARSE)
        idx = SparseTextFieldIndex(cfg)
        idx.add_documents(_DOC_IDS, _DOCS, index_dir=tmp_path / "sparse_content")

        results = idx.search("apple", k=3)

        assert len(results.items) > 0
        top_id = results.items[0].doc_id
        assert top_id == "doc_0"

    def test_search_before_add_returns_empty(self) -> None:
        cfg = FieldConfig("content", TokenizerType.SPLIT, RepresentationType.SPARSE)
        idx = SparseTextFieldIndex(cfg)

        results = idx.search("apple", k=3)

        assert results.items == []

    def test_search_returns_relevant_doc(self, tmp_path: Path) -> None:
        cfg = FieldConfig("content", TokenizerType.SPLIT, RepresentationType.SPARSE)
        idx = SparseTextFieldIndex(cfg)
        idx.add_documents(_DOC_IDS, _DOCS, index_dir=tmp_path / "sparse_content")

        results = idx.search("banana guide", k=2)

        doc_ids_returned = {item.doc_id for item in results.items}
        assert "doc_1" in doc_ids_returned

    def test_index_dir_used_for_persistence(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "bm25_field"
        cfg = FieldConfig("content", TokenizerType.SPLIT, RepresentationType.SPARSE)
        idx = SparseTextFieldIndex(cfg)
        idx.add_documents(_DOC_IDS, _DOCS, index_dir=index_dir)

        assert index_dir.exists()


class TestDenseTextFieldIndex:
    def _benchmark_cfg(self) -> FieldConfig:
        return FieldConfig(
            "content",
            TokenizerType.SPLIT,
            RepresentationType.DENSE,
            usage=IndexUsage.BENCHMARK,
        )

    def test_add_documents_in_memory(self) -> None:
        cfg = self._benchmark_cfg()
        idx = DenseTextFieldIndex(cfg, _StubEncoder())
        idx.add_documents(_DOC_IDS, _DOCS)

        assert idx.indexer.index is not None
        assert idx.indexer.index.ntotal == len(_DOC_IDS)

    def test_search_returns_closest_doc(self) -> None:
        cfg = self._benchmark_cfg()
        idx = DenseTextFieldIndex(cfg, _StubEncoder())
        idx.add_documents(_DOC_IDS, _DOCS)

        # Query starting with "a" → one-hot at same dim as "apple pie recipe"
        results = idx.search("apple query", k=3)

        assert len(results.items) > 0
        assert results.items[0].doc_id == "doc_0"

    def test_search_empty_index_returns_empty(self) -> None:
        cfg = self._benchmark_cfg()
        idx = DenseTextFieldIndex(cfg, _StubEncoder())

        results = idx.search("anything", k=5)

        assert results.items == []

    def test_scores_bounded(self) -> None:
        cfg = self._benchmark_cfg()
        idx = DenseTextFieldIndex(cfg, _StubEncoder())
        idx.add_documents(_DOC_IDS, _DOCS)

        results = idx.search("cherry recipe", k=3)

        for item in results.items:
            assert -1.01 <= item.score <= 1.01


class TestHybridUnion:
    """Verify that sparse + dense union covers expected doc_ids."""

    def test_union_contains_answer_doc(self, tmp_path: Path) -> None:
        sparse_cfg = FieldConfig("content", TokenizerType.SPLIT, RepresentationType.SPARSE)
        sparse = SparseTextFieldIndex(sparse_cfg)
        sparse.add_documents(_DOC_IDS, _DOCS, index_dir=tmp_path / "sparse")

        dense_cfg = FieldConfig(
            "content",
            TokenizerType.SPLIT,
            RepresentationType.DENSE,
            usage=IndexUsage.BENCHMARK,
        )
        dense = DenseTextFieldIndex(dense_cfg, _StubEncoder())
        dense.add_documents(_DOC_IDS, _DOCS)

        k = 3
        union: dict[str, float] = {}
        for item in sparse.search("banana guide", k).items:
            union[item.doc_id] = union.get(item.doc_id, 0.0) + item.score
        for item in dense.search("banana smoothie", k).items:
            union[item.doc_id] = union.get(item.doc_id, 0.0) + item.score

        assert "doc_1" in union

    def test_union_non_empty_for_valid_query(self, tmp_path: Path) -> None:
        sparse_cfg = FieldConfig("content", TokenizerType.SPLIT, RepresentationType.SPARSE)
        sparse = SparseTextFieldIndex(sparse_cfg)
        sparse.add_documents(_DOC_IDS, _DOCS, index_dir=tmp_path / "sparse")

        dense_cfg = FieldConfig(
            "content",
            TokenizerType.SPLIT,
            RepresentationType.DENSE,
            usage=IndexUsage.BENCHMARK,
        )
        dense = DenseTextFieldIndex(dense_cfg, _StubEncoder())
        dense.add_documents(_DOC_IDS, _DOCS)

        union: set[str] = set()
        for item in sparse.search("cherry tutorial", 3).items:
            union.add(item.doc_id)
        for item in dense.search("cherry tutorial", 3).items:
            union.add(item.doc_id)

        assert len(union) > 0
