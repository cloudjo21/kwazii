"""Unit tests for Phase 6 STaRK-Prime training and shortlist cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from asmr.datasets.stark_prime.shortlist_cache import (
    load_shortlist_cache,
    save_shortlist_cache,
    shortlist_cache_path,
)
from asmr.datasets.stark_prime.torch_dataset import StarkRankingExample
from asmr.train.stark_prime_training import (
    TrainingPhase,
    check_gate_90,
    resolve_training_phase,
)


class TestGate90:
    def test_pass_when_all_metrics_above_threshold(self) -> None:
        ratio = {"hit@1": 0.91, "recall@20": 0.92, "mrr": 0.90}
        passed, checks = check_gate_90(ratio)
        assert passed
        assert all(checks.values())

    def test_fail_when_any_metric_below_threshold(self) -> None:
        ratio = {"hit@1": 0.91, "recall@20": 0.80, "mrr": 0.95}
        passed, checks = check_gate_90(ratio)
        assert not passed
        assert checks["recall@20"] is False


class TestTrainingPhase:
    def test_jina_downgrades_joint_to_head_only(self) -> None:
        encoder = MagicMock()
        encoder.name = "jinavera_1024"
        encoder.supports_joint_training.return_value = False
        phase = resolve_training_phase(TrainingPhase.JOINT, encoder)
        assert phase == TrainingPhase.HEAD_ONLY

    def test_jina_lora_keeps_joint(self) -> None:
        encoder = MagicMock()
        encoder.name = "jinavera_lora_1024_stark_prime"
        encoder.supports_joint_training.return_value = True
        phase = resolve_training_phase(TrainingPhase.JOINT, encoder)
        assert phase == TrainingPhase.JOINT

    def test_contriever_keeps_joint(self) -> None:
        encoder = MagicMock()
        encoder.name = "contriever_facebook_contriever-msmarco"
        encoder.supports_joint_training.return_value = True
        phase = resolve_training_phase(TrainingPhase.JOINT, encoder)
        assert phase == TrainingPhase.JOINT


class TestShortlistCache:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        ex = StarkRankingExample(
            query_text="test query",
            doc_ids=["d1", "d2"],
            scores=np.ones((2, 2, 2), dtype=np.float32),
            relevance=np.array([1.0, 0.0], dtype=np.float32),
            field_mask=np.ones((2, 2), dtype=bool),
        )
        from asmr.datasets.stark_prime.loader import PrimeQuery

        queries = [PrimeQuery(query_id=1, query="test query", answer_ids=["d1"])]
        save_shortlist_cache(
            [ex],
            queries,
            tmp_path,
            "train",
            shortlist_k=50,
            encoder_name="contriever_test",
        )
        path = shortlist_cache_path(
            tmp_path,
            "train",
            50,
            "contriever_test",
        )
        assert path.exists()
        loaded = load_shortlist_cache(
            tmp_path,
            "train",
            50,
            "contriever_test",
        )
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].doc_ids == ["d1", "d2"]
        assert loaded[0].scores.shape == (2, 2, 2)

    def test_load_returns_none_on_encoder_mismatch(self, tmp_path: Path) -> None:
        ex = StarkRankingExample(
            query_text="q",
            doc_ids=["d1"],
            scores=np.ones((1, 2, 1), dtype=np.float32),
            relevance=np.array([1.0], dtype=np.float32),
        )
        from asmr.datasets.stark_prime.loader import PrimeQuery

        save_shortlist_cache(
            [ex],
            [PrimeQuery(query_id=1, query="q", answer_ids=["d1"])],
            tmp_path,
            "train",
            shortlist_k=10,
            encoder_name="enc_a",
        )
        assert load_shortlist_cache(tmp_path, "train", 10, "enc_b") is None
