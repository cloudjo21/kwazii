"""Unit tests for STaRK-Prime v2.1 disk index cache and equivalence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pytest

from asmr.evaluation.stark_prime_disk_index_v2 import (
    FieldIndexCache,
    PrimeDiskIndexStore,
    ShortlistTiming,
    _dense_topk_faiss,
    _dense_topk_memmap,
    build_dense_faiss_field,
)

_DATA_ROOT = Path(__file__).resolve().parents[4] / "data" / "stark_prime"
_INDEX_DIR = _DATA_ROOT / "index" / "prime"
_HAS_INDEX = (_INDEX_DIR / "manifest.json").exists()


class TestDenseTopkEquivalence:
    """Memmap dot vs FAISS IndexFlatIP must yield identical top-k doc ids."""

    def test_memmap_vs_faiss_topk_match(self, tmp_path: Path) -> None:
        """Top-k doc_id sets match for L2-normalized vectors."""
        rng = np.random.default_rng(42)
        n_docs, dim, k = 200, 32, 10
        doc_ids = [f"d{i}" for i in range(n_docs)]
        vectors = rng.standard_normal((n_docs, dim)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)

        memmap_path = tmp_path / "dense.f32.mmap"
        faiss_path = tmp_path / "dense.faiss"
        mm = np.memmap(
            memmap_path,
            dtype=np.float32,
            mode="w+",
            shape=(n_docs, dim),
        )
        mm[:] = vectors
        mm.flush()
        build_dense_faiss_field(
            memmap_path,
            faiss_path,
            n_docs=n_docs,
            dim=dim,
        )

        q = rng.standard_normal(dim).astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        mmap_hits = _dense_topk_memmap(mm, doc_ids, q, k)
        index = faiss.read_index(str(faiss_path))
        faiss_hits = _dense_topk_faiss(index, doc_ids, q, k)

        mmap_ids = {doc_id for doc_id, _ in mmap_hits}
        faiss_ids = {doc_id for doc_id, _ in faiss_hits}
        assert mmap_ids == faiss_ids

        mmap_scores = {d: s for d, s in mmap_hits}
        faiss_scores = {d: s for d, s in faiss_hits}
        for doc_id in mmap_ids:
            assert mmap_scores[doc_id] == pytest.approx(
                faiss_scores[doc_id],
                abs=1e-4,
            )


class TestFieldIndexCache:
    """FieldIndexCache loads each field at most once."""

    def test_get_sparse_returns_same_instance(self, tmp_path: Path) -> None:
        """Second get_sparse call returns cached BM25 index."""
        field = "name"
        slug = "name"
        sparse_dir = tmp_path / slug / "sparse"
        sparse_dir.mkdir(parents=True)
        (sparse_dir / "metadata.json").write_text("{}")
        (sparse_dir / "vocab.trie").write_text("")

        cache = FieldIndexCache(
            tmp_path,
            [field],
            num_docs=2,
            embedding_dim=4,
            dense_backend="memmap",
        )
        mock_index = object()
        with patch(
            "asmr.evaluation.stark_prime_disk_index_v2._load_sparse_field",
            return_value=mock_index,
        ) as load_mock:
            first = cache.get_sparse(field)
            second = cache.get_sparse(field)
        assert first is second
        load_mock.assert_called_once()


@pytest.mark.skipif(not _HAS_INDEX, reason="STaRK-Prime index not built")
class TestPrimeDiskIndexStoreEquivalence:
    """Shortlist outputs must match across v2.1 code paths."""

    @pytest.fixture
    def store(self) -> PrimeDiskIndexStore:
        return PrimeDiskIndexStore(_INDEX_DIR)

    @pytest.fixture
    def sample_queries(self) -> list[tuple[str, np.ndarray]]:
        from asmr.datasets.stark_prime.loader import load_prime_queries
        from asmr.train.query_encoder import HfQueryEncoder

        encoder = HfQueryEncoder(
            model_name="facebook/contriever-msmarco",
            device="cpu",
        )
        queries = load_prime_queries(_DATA_ROOT, "test")[:10]
        return [(q.query, encoder.encode([q.query])[0].numpy()) for q in queries]

    def test_sequential_vs_parallel_shortlist(
        self,
        store: PrimeDiskIndexStore,
        sample_queries: list[tuple[str, np.ndarray]],
    ) -> None:
        """Parallel and sequential shortlists produce identical doc unions."""
        k = 20
        for query_text, q_emb in sample_queries:
            seq_ids, seq_scores, seq_mask = store.shortlist_hybrid(
                query_text,
                q_emb,
                k,
            )
            par_ids, par_scores, par_mask = store.shortlist_hybrid_dispatch(
                query_text,
                q_emb,
                k,
                parallel=True,
            )
            assert seq_ids == par_ids
            assert seq_scores.shape == par_scores.shape
            np.testing.assert_allclose(seq_scores, par_scores, rtol=0, atol=1e-5)
            np.testing.assert_array_equal(seq_mask, par_mask)

    def test_field_cache_reuses_sparse(
        self,
        store: PrimeDiskIndexStore,
    ) -> None:
        """FieldIndexCache returns same sparse handle on repeated access."""
        field = store.field_names[0]
        first = store.field_cache.get_sparse(field)
        second = store.field_cache.get_sparse(field)
        assert first is second

    def test_timing_populated(
        self,
        store: PrimeDiskIndexStore,
        sample_queries: list[tuple[str, np.ndarray]],
    ) -> None:
        """ShortlistTiming records sparse/dense stage milliseconds."""
        query_text, q_emb = sample_queries[0]
        timing = ShortlistTiming()
        store.shortlist_hybrid(
            query_text,
            q_emb,
            10,
            timing=timing,
        )
        assert timing.total_ms > 0
        assert len(timing.field_timings) == len(store.field_names)
