"""Test fixtures and module mocks for asmr.match unit tests.

Mocks heavy/unavailable modules before any imports so that the full
asmr package can be imported without GPU, kiwipiepy, or muvfde.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Block import of modules not installed in the test environment.
# These are set BEFORE any asmr.* imports so that the lazy-import chain
# inside indexer.py / ranker.py does not trigger ModuleNotFoundError.
# ---------------------------------------------------------------------------
for _mod in ("kiwipiepy",):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# muvfde needs its submodule mocked too (fde.config imports muvfde.muvfde_ext)
if "muvfde" not in sys.modules:
    _muvfde = MagicMock()
    _muvfde_ext = MagicMock()
    _muvfde.muvfde_ext = _muvfde_ext
    sys.modules["muvfde"] = _muvfde
    sys.modules["muvfde.muvfde_ext"] = _muvfde_ext


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_encoder() -> MagicMock:
    """Minimal BaseFdeEncoder mock that returns unit numpy vectors."""
    import numpy as np

    enc = MagicMock()
    enc.encode_image.return_value = np.ones((1, 64), dtype=np.float32)
    enc.encode_text.return_value = np.ones((1, 64), dtype=np.float32)
    return enc


@pytest.fixture()
def image_urls() -> list[str]:
    """20 placeholder image URLs (no network access needed in unit tests)."""
    return [f"http://example.com/img_{i}.jpg" for i in range(20)]
