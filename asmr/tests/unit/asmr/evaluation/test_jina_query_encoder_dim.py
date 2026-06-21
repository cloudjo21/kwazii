"""Tests for JinaQueryEncoder FDE projection vs legacy prefix truncate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from asmr.evaluation.query_encoders import JinaQueryEncoder, _validate_mrl_dim


class TestJinaQueryEncoderDim:
    def test_validate_mrl_dim_rejects_invalid(self) -> None:
        """768 is not a supported MRL dimension."""
        with pytest.raises(ValueError, match="768"):
            _validate_mrl_dim(768, label="truncate_dim")

    @patch("fde.models.jinavera.Jinavera")
    def test_native_fde_skips_prefix_truncate(
        self,
        mock_jina_cls: MagicMock,
    ) -> None:
        """When fde_output_dim is set, vectors are not prefix-sliced."""
        mock_jina = MagicMock()
        mock_jina.embedding_dim = 512
        mock_jina.encode_text.return_value = np.ones((1, 512), dtype=np.float32)
        mock_jina_cls.return_value = mock_jina

        enc = JinaQueryEncoder(fde_output_dim=512, truncate_dim=512)
        out = enc.encode_text(["hello"])
        assert out.shape == (1, 512)
        mock_jina_cls.assert_called_once()
        assert mock_jina_cls.call_args.kwargs["fde_output_dim"] == 512

    @patch("fde.models.jinavera.Jinavera")
    def test_legacy_prefix_truncate_when_no_fde_output_dim(
        self,
        mock_jina_cls: MagicMock,
    ) -> None:
        """Legacy mode slices 10240-d FDE down to truncate_dim."""
        mock_jina = MagicMock()
        mock_jina.embedding_dim = 10240
        full = np.arange(10240, dtype=np.float32).reshape(1, 10240)
        mock_jina.encode_text.return_value = full
        mock_jina_cls.return_value = mock_jina

        enc = JinaQueryEncoder(fde_output_dim=None, truncate_dim=512)
        out = enc.encode_text(["hello"])
        assert out.shape == (1, 512)
        assert "fde_output_dim" not in mock_jina_cls.call_args.kwargs
        norm = np.linalg.norm(out[0])
        assert norm == pytest.approx(1.0, rel=1e-4)
