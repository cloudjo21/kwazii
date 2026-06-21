"""Tests for ADR-003 FDE final_projection_dimension support."""

from __future__ import annotations

import numpy as np

from fde.config import (
    FdeConfig,
    PromptType,
    MRL_DIMENSIONS,
    fde_config_fingerprint,
)


class TestFdeOutputDim:
    def test_apply_with_fde_output_dim_produces_target_size(self) -> None:
        """final_projection_dimension sets native FDE output size."""
        import muvfde

        cfg = FdeConfig.apply_with_fde_output_dim(PromptType.QUERY, 512)
        multivector = np.random.randn(8, 64).astype(np.float32)
        out = muvfde.generate_fixed_dimensional_encoding(multivector, cfg)
        assert out.shape == (512,)

    def test_fingerprint_stable_for_same_dim(self) -> None:
        """fde_config_hash is deterministic per output dim."""
        a = fde_config_fingerprint(512)
        b = fde_config_fingerprint(512)
        c = fde_config_fingerprint(1024)
        assert a == b
        assert a != c

    def test_mrl_dimensions_include_v3_defaults(self) -> None:
        """Default ablation dims are in MRL set."""
        assert 512 in MRL_DIMENSIONS
        assert 1024 in MRL_DIMENSIONS
