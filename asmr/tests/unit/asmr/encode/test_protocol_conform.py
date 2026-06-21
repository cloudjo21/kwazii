"""Protocol conformance tests for unified text encoder API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from asmr.encode.protocol import (
    NamedTextEncoderProtocol,
    TextEncoderProtocol,
    TrainableTextEncoderProtocol,
)
from asmr.train.query_encoder import HfQueryEncoder
from fde.config import PromptType


class TestHfQueryEncoderProtocol:
    def test_is_named_text_encoder_protocol(self) -> None:
        """HfQueryEncoder structurally conforms to NamedTextEncoderProtocol."""
        enc = HfQueryEncoder(model_name="facebook/contriever-msmarco", device="cpu")
        assert isinstance(enc, NamedTextEncoderProtocol)
        assert isinstance(enc, TrainableTextEncoderProtocol)

    def test_encode_text_matches_legacy_encode(self) -> None:
        """encode_text output matches deprecated encode().numpy()."""
        enc = HfQueryEncoder(model_name="facebook/contriever-msmarco", device="cpu")
        texts = ["hello world", "second query"]
        legacy = enc.encode(texts).numpy()
        canonical = enc.encode_text(texts, PromptType.QUERY)
        np.testing.assert_allclose(canonical, legacy, rtol=1e-5, atol=1e-5)

    def test_prompt_type_noop_for_contriever(self) -> None:
        """Contriever query/passage pooling is identical."""
        enc = HfQueryEncoder(model_name="facebook/contriever-msmarco", device="cpu")
        texts = ["query text"]
        q = enc.encode_text(texts, PromptType.QUERY)
        p = enc.encode_text(texts, PromptType.PASSAGE)
        np.testing.assert_allclose(q, p, rtol=1e-6, atol=1e-6)

    def test_l2_normalized_rows(self) -> None:
        """Contriever encode_text returns unit-norm rows."""
        enc = HfQueryEncoder(model_name="facebook/contriever-msmarco", device="cpu")
        emb = enc.encode_text(["normalize me"], PromptType.QUERY)
        norm = np.linalg.norm(emb[0])
        assert norm == pytest.approx(1.0, rel=1e-4)

    def test_encode_trainable_has_grad(self) -> None:
        """Backward reaches HF model parameters."""
        enc = HfQueryEncoder(model_name="facebook/contriever-msmarco", device="cpu")
        emb = enc.encode_trainable(["grad check"], PromptType.QUERY)
        loss = emb.sum()
        loss.backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert grads


class TestBaseFdeEncoderProtocol:
    @patch("fde.base.muvfde.generate_fixed_dimensional_encoding")
    def test_jinavera_conforms_text_encoder(
        self,
        mock_fde: MagicMock,
    ) -> None:
        """BaseFdeEncoder subclass satisfies TextEncoderProtocol."""
        mock_fde.return_value = np.ones(128, dtype=np.float32)

        class _StubEncoder:
            encoder_model_path = "stub"
            fde_configs = {
                PromptType.QUERY: MagicMock(),
                PromptType.PASSAGE: MagicMock(),
            }

            @property
            def embedding_dim(self) -> int:
                return 128

            def _encode_texts_to_multivector(self, texts, prompt_type):
                del prompt_type
                return [torch.ones(4, 8)]

            def encode_text(self, texts, prompt_type=PromptType.QUERY):
                if isinstance(texts, str):
                    texts = [texts]
                del texts
                return np.ones((1, 128), dtype=np.float32)

        stub = _StubEncoder()
        assert isinstance(stub, TextEncoderProtocol)
