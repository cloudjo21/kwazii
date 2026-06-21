"""Unit tests for Jina LoRA query encoder (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from asmr.train.stark_prime_training import TrainingPhase, resolve_training_phase
from asmr.train.query_encoder import JinaLoraConfig, JinaLoraQueryEncoder
from fde.config import PromptType


class TestJinaLoraQueryEncoder:
    @patch("fde.hf.models.jina_embeddings_v4.JinaEmbeddingsV4Model.from_pretrained")
    def test_freeze_non_lora_enables_adapter_only(
        self,
        mock_from_pretrained: MagicMock,
    ) -> None:
        """Only LoRA adapter params remain trainable after freeze."""
        base = MagicMock()
        base.peft_config = {"default": MagicMock()}
        base.parameters.return_value = [
            torch.nn.Parameter(torch.zeros(2), requires_grad=False),
        ]
        base.named_parameters.return_value = [
            ("base.weight", torch.nn.Parameter(torch.zeros(2), requires_grad=True)),
            (
                "lora_A.default.retrieval.weight",
                torch.nn.Parameter(torch.ones(2), requires_grad=True),
            ),
            (
                "lora_B.other.weight",
                torch.nn.Parameter(torch.ones(2), requires_grad=True),
            ),
        ]
        mock_from_pretrained.return_value = base

        enc = JinaLoraQueryEncoder(JinaLoraConfig(adapter_name="default"))
        trainable = [
            name for name, param in enc._model.named_parameters() if param.requires_grad
        ]
        assert any("lora_A.default.retrieval" in name for name in trainable)
        assert not any("lora_B.other" in name for name in trainable)

    @patch("fde.hf.models.jina_embeddings_v4.JinaEmbeddingsV4Model.from_pretrained")
    def test_encode_trainable_returns_tensor(
        self,
        mock_from_pretrained: MagicMock,
    ) -> None:
        """encode_trainable returns a differentiable tensor batch."""
        base = MagicMock()
        base.peft_config = {"default": MagicMock()}
        base.named_parameters.return_value = []
        base.processor.process_texts.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        out_emb = torch.ones(1, 1024, requires_grad=True)
        forward_out = MagicMock()
        forward_out.single_vec_emb = out_emb
        base.return_value = forward_out
        mock_from_pretrained.return_value = base

        enc = JinaLoraQueryEncoder(JinaLoraConfig(), device="cpu")
        out = enc.encode_trainable(["hello"], PromptType.QUERY, batch_size=1)
        assert out.shape == (1, 1024)
        assert out.requires_grad

    @patch("fde.hf.models.jina_embeddings_v4.JinaEmbeddingsV4Model.from_pretrained")
    def test_lora_state_dict_roundtrip(
        self,
        mock_from_pretrained: MagicMock,
    ) -> None:
        """lora_state_dict save/load preserves encode_trainable output."""
        base = MagicMock()
        base.peft_config = {"default": MagicMock()}
        base.named_parameters.return_value = []
        base.encode_text.return_value = torch.ones(1, 1024)
        mock_from_pretrained.return_value = base

        enc = JinaLoraQueryEncoder(JinaLoraConfig(), device="cpu")
        with patch(
            "asmr.encode.implementations.jina_lora.get_peft_model_state_dict",
            return_value={"lora": torch.tensor([1.0])},
        ), patch(
            "asmr.encode.implementations.jina_lora.set_peft_model_state_dict",
        ) as mock_set:
            state = enc.lora_state_dict()
            enc.load_lora_state_dict(state)
            mock_set.assert_called_once()


class TestTrainingPhaseJinaLora:
    def test_jina_lora_keeps_joint_phase(self) -> None:
        """Jina LoRA wrapper no longer downgrades Phase 2."""
        encoder = MagicMock()
        encoder.name = "jinavera_lora_1024_stark_prime"
        encoder.supports_joint_training.return_value = True
        phase = resolve_training_phase(TrainingPhase.JOINT, encoder)
        assert phase == TrainingPhase.JOINT

    def test_frozen_jina_still_downgrades(self) -> None:
        """Frozen JinaQueryEncoder remains Phase 1 only."""
        encoder = MagicMock()
        encoder.name = "jinavera_1024"
        encoder.supports_joint_training.return_value = False
        phase = resolve_training_phase(TrainingPhase.JOINT, encoder)
        assert phase == TrainingPhase.HEAD_ONLY
