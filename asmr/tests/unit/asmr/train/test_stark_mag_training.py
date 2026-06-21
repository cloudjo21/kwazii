"""Unit tests for stark_mag_training and related MAG training infrastructure."""

import pickle
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fde.config import PromptType

from asmr.datasets.stark_mag.loader import MagQuery
# Import from asmr.train first so train/__init__ loads train.data before torch_dataset,
# avoiding the circular import that occurs when torch_dataset is loaded before train.
from asmr.train import StarkRankingExample
from asmr.train.config import TrainMfarConfig
from asmr.train.stark_mag_training import (
    GATE_RATIO_90_MAG,
    build_mag_training_examples,
    check_gate_90_mag,
    train_mfar_mag,
)
from asmr.datasets.stark_mag.shortlist_cache import (
    load_shortlist_cache,
    save_shortlist_cache,
    shortlist_cache_path,
)

_DIM = 16
_N_FIELDS = 5
_N_DOCS = 6


def _make_query(qid: int, answer_id: str) -> MagQuery:
    return MagQuery(query_id=qid, query=f"query {qid}", answer_ids=[answer_id])


def _make_example(n_docs: int = _N_DOCS) -> StarkRankingExample:
    scores = np.random.rand(_N_FIELDS, 2, n_docs).astype(np.float32)
    relevance = np.zeros(n_docs, dtype=np.float32)
    relevance[0] = 1.0
    mask = np.ones((_N_FIELDS, n_docs), dtype=bool)
    return StarkRankingExample(
        query_text="test query",
        doc_ids=[f"doc_{i}" for i in range(n_docs)],
        scores=scores,
        relevance=relevance,
        field_mask=mask,
    )


class _StubEncoder:
    @property
    def embedding_dim(self) -> int:
        return _DIM

    def encode_text(self, texts, prompt_type, *, batch_size=32):
        emb = np.zeros((len(texts), _DIM), dtype=np.float32)
        emb[:, 0] = 1.0
        return emb


def _make_mock_indexes(n_docs: int = _N_DOCS) -> MagicMock:
    """Returns a MagFieldIndexes mock with shortlist_hybrid returning fixed data."""
    doc_ids = [f"doc_{i}" for i in range(n_docs)]
    scores = np.random.rand(_N_FIELDS, 2, n_docs).astype(np.float32)
    mask = np.ones((_N_FIELDS, n_docs), dtype=bool)

    mock = MagicMock()
    mock.shortlist_hybrid.return_value = (doc_ids, scores, mask)
    return mock


# ── shortlist_cache ──────────────────────────────────────────────────────────

class TestShortlistCache:
    def test_roundtrip(self, tmp_path: Path) -> None:
        examples = [_make_example(), _make_example()]
        save_shortlist_cache(examples, tmp_path, "train", 100, "contriever")
        loaded = load_shortlist_cache(tmp_path, "train", 100, "contriever")
        assert loaded is not None
        assert len(loaded) == 2

    def test_cache_path_slug(self, tmp_path: Path) -> None:
        path = shortlist_cache_path(tmp_path, "train", 100, "facebook/contriever-msmarco")
        assert "facebook_contriever-msmarco" in path.name

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        result = load_shortlist_cache(tmp_path, "train", 100, "no_such_encoder")
        assert result is None

    def test_version_mismatch_returns_none(self, tmp_path: Path) -> None:
        path = shortlist_cache_path(tmp_path, "train", 100, "enc")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"version": 999, "examples": []}, f)
        result = load_shortlist_cache(tmp_path, "train", 100, "enc")
        assert result is None

    def test_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        path = shortlist_cache_path(tmp_path, "train", 100, "enc")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-a-pickle")
        result = load_shortlist_cache(tmp_path, "train", 100, "enc")
        assert result is None


# ── build_mag_training_examples ─────────────────────────────────────────────

class TestBuildMagTrainingExamples:
    def test_builds_examples_no_cache(self) -> None:
        queries = [_make_query(i, f"doc_{i}") for i in range(3)]
        indexes = _make_mock_indexes()
        encoder = _StubEncoder()

        examples = build_mag_training_examples(
            queries, indexes, encoder, shortlist_k=5
        )

        assert len(examples) == 3
        assert indexes.shortlist_hybrid.call_count == 3

    def test_cache_hit_skips_index(self, tmp_path: Path) -> None:
        cached = [_make_example()]
        save_shortlist_cache(cached, tmp_path, "train", 5, "enc")

        queries = [_make_query(0, "doc_0")]
        indexes = _make_mock_indexes()
        encoder = _StubEncoder()

        examples = build_mag_training_examples(
            queries, indexes, encoder, shortlist_k=5,
            cache_dir=tmp_path, encoder_name="enc"
        )

        assert len(examples) == 1
        indexes.shortlist_hybrid.assert_not_called()

    def test_cache_write_on_miss(self, tmp_path: Path) -> None:
        queries = [_make_query(0, "doc_0")]
        indexes = _make_mock_indexes()
        encoder = _StubEncoder()

        build_mag_training_examples(
            queries, indexes, encoder, shortlist_k=5,
            cache_dir=tmp_path, encoder_name="enc"
        )

        path = shortlist_cache_path(tmp_path, "train", 5, "enc")
        assert path.exists()

    def test_rebuild_cache_ignores_existing(self, tmp_path: Path) -> None:
        stale = [_make_example(), _make_example()]
        save_shortlist_cache(stale, tmp_path, "train", 5, "enc")

        queries = [_make_query(0, "doc_0")]
        indexes = _make_mock_indexes()
        encoder = _StubEncoder()

        examples = build_mag_training_examples(
            queries, indexes, encoder, shortlist_k=5,
            cache_dir=tmp_path, encoder_name="enc", rebuild_cache=True
        )

        assert len(examples) == 1
        indexes.shortlist_hybrid.assert_called_once()

    def test_empty_shortlist_skipped(self) -> None:
        queries = [_make_query(0, "doc_0")]
        indexes = MagicMock()
        indexes.shortlist_hybrid.return_value = (
            [], np.zeros((_N_FIELDS, 2, 0)), np.zeros((_N_FIELDS, 0), dtype=bool)
        )
        encoder = _StubEncoder()

        examples = build_mag_training_examples(
            queries, indexes, encoder, shortlist_k=5
        )

        assert examples == []


# ── train_mfar_mag ───────────────────────────────────────────────────────────

class TestTrainMfarMag:
    def test_returns_adapter(self) -> None:
        import torch
        from asmr.train.aggregation import MFARFieldAdapter

        queries = [_make_query(i, f"doc_{i}") for i in range(4)]
        indexes = _make_mock_indexes()
        encoder = _StubEncoder()
        cfg = TrainMfarConfig(shortlist_k=5, epochs=1, adapter_lr=1e-3)
        device = torch.device("cpu")

        adapter = train_mfar_mag(queries, indexes, encoder, cfg, device)

        assert isinstance(adapter, MFARFieldAdapter)

    def test_adapter_parameters_updated(self) -> None:
        import torch
        from asmr.train.aggregation import MFARFieldAdapter

        queries = [_make_query(i, f"doc_{i}") for i in range(3)]
        indexes = _make_mock_indexes()
        encoder = _StubEncoder()
        cfg = TrainMfarConfig(shortlist_k=5, epochs=2, adapter_lr=1e-2)
        device = torch.device("cpu")

        adapter_before = MFARFieldAdapter(encoder.embedding_dim, _N_FIELDS, 2)
        w_before = adapter_before._weight_logits.weight.clone()

        adapter = train_mfar_mag(queries, indexes, encoder, cfg, device)
        w_after = adapter._weight_logits.weight

        assert not torch.allclose(w_before, w_after), "weights should change after training"


# ── check_gate_90_mag ────────────────────────────────────────────────────────

class TestCheckGate90Mag:
    def test_all_pass(self) -> None:
        ratio = {"hit@1": 0.91, "recall@20": 0.95, "mrr": 0.92}
        passed, checks = check_gate_90_mag(ratio)
        assert passed
        assert all(checks.values())

    def test_one_fail(self) -> None:
        ratio = {"hit@1": 0.89, "recall@20": 0.95, "mrr": 0.92}
        passed, checks = check_gate_90_mag(ratio)
        assert not passed
        assert not checks["hit@1"]

    def test_exact_threshold_passes(self) -> None:
        ratio = {k: 0.90 for k in GATE_RATIO_90_MAG}
        passed, _ = check_gate_90_mag(ratio)
        assert passed

    def test_missing_key_fails(self) -> None:
        passed, checks = check_gate_90_mag({})
        assert not passed
