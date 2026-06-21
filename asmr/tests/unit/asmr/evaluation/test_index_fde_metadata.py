"""Tests for index manifest FDE metadata (ADR-003)."""

from __future__ import annotations

from unittest.mock import MagicMock

from asmr.evaluation.stark_prime_disk_index_v2 import _index_metadata_for_encoder


class TestIndexMetadata:
    def test_includes_fde_fields_for_jina_encoder(self) -> None:
        """Jina encoders with fde_output_dim add manifest extras."""
        enc = MagicMock()
        enc.fde_output_dim = 512
        meta = _index_metadata_for_encoder(enc)
        assert meta["fde_output_dim"] == 512
        assert "fde_config_hash" in meta
        assert len(str(meta["fde_config_hash"])) == 16

    def test_empty_for_contriever(self) -> None:
        """Non-Jina encoders omit FDE metadata."""
        enc = MagicMock(spec=[])
        meta = _index_metadata_for_encoder(enc)
        assert meta == {}
