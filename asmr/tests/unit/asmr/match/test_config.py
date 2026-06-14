"""Unit tests for asmr.match.config."""

import pytest

from asmr.match.config import RankerConfig


class TestRankerConfig:
    """Tests for RankerConfig dataclass defaults and custom values."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        cfg = RankerConfig()
        assert cfg.field_name == "image"
        assert cfg.top_k == 5
        assert cfg.image_load_timeout_s == 10.0
        assert cfg.device == "cpu"

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        cfg = RankerConfig(
            field_name="thumb",
            top_k=10,
            image_load_timeout_s=5.0,
            device="cuda",
        )
        assert cfg.field_name == "thumb"
        assert cfg.top_k == 10
        assert cfg.image_load_timeout_s == 5.0
        assert cfg.device == "cuda"

    @pytest.mark.parametrize(
        "top_k",
        [1, 5, 20, 100],
    )
    def test_top_k_variants(self, top_k: int) -> None:
        """top_k accepts any positive integer."""
        cfg = RankerConfig(top_k=top_k)
        assert cfg.top_k == top_k

    def test_is_dataclass_instance(self) -> None:
        """RankerConfig is a dataclass (supports equality)."""
        assert RankerConfig() == RankerConfig()
        assert RankerConfig(top_k=3) != RankerConfig(top_k=5)
