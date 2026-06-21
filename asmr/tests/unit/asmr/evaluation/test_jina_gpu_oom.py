"""Tests for Jina CUDA OOM recovery helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from asmr.evaluation.jina_gpu_oom import (
    JINA_OOM_ENCODE_BATCH,
    is_cuda_oom,
)
from asmr.evaluation.query_encoders import JinaQueryEncoder


class TestIsCudaOom:
    def test_detects_torch_cuda_oom(self) -> None:
        """torch.cuda.OutOfMemoryError is recognized."""
        assert is_cuda_oom(torch.cuda.OutOfMemoryError("boom"))

    def test_detects_message_substring(self) -> None:
        """Generic exceptions with OOM text are recognized."""
        assert is_cuda_oom(RuntimeError("CUDA error: out of memory"))

    def test_rejects_unrelated_error(self) -> None:
        """Non-OOM errors are not treated as CUDA OOM."""
        assert not is_cuda_oom(ValueError("bad dim"))


class TestJinaQueryEncoderOomRecovery:
    @patch("fde.models.jinavera.Jinavera")
    def test_retries_with_batch_four_then_reload(
        self,
        mock_jina_cls: MagicMock,
    ) -> None:
        """OOM path reduces batch to 4 and reloads the model."""
        import numpy as np

        mock_jina = MagicMock()
        mock_jina.embedding_dim = 512
        ok = np.ones((1, 512), dtype=np.float32)
        mock_jina.encode_text.side_effect = [
            torch.cuda.OutOfMemoryError("oom1"),
            torch.cuda.OutOfMemoryError("oom2"),
            ok,
        ]
        mock_jina_cls.return_value = mock_jina

        enc = JinaQueryEncoder(fde_output_dim=512, truncate_dim=512)
        out = enc.encode_text(["hello"], batch_size=8)

        assert out.shape == (1, 512)
        assert enc.jina_fieldwise_gpu is True
        assert mock_jina.encode_text.call_args_list[-1].kwargs["encode_batch_size"] == (
            JINA_OOM_ENCODE_BATCH
        )
        mock_jina.unload_model.assert_called_once()
        mock_jina.ensure_model_loaded.assert_called_once()
