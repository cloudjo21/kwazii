"""GPU integration tests for asmr.match.

These tests exercise the full inference pipeline on a real GPU with a real
encoder and FAISS index.  They are unconditionally skipped unless:

  1. torch.cuda.is_available() == True
  2. ASMR_INTEGRATION_TESTS=1 environment variable is set

To run on a GPU machine::

    ASMR_INTEGRATION_TESTS=1 pytest asmr/tests/integration/test_image_ranker_gpu.py -v

The tests assume a JinaVera-compatible encoder is available at
ENCODER_MODEL_PATH (defaults to "jinaai/jina-embeddings-v4").
"""

from __future__ import annotations

import os

import numpy as np
import pytest

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

_INTEGRATION_ENABLED = os.getenv("ASMR_INTEGRATION_TESTS", "0") == "1"
_SKIP_REASON = (
    "GPU integration tests require CUDA and ASMR_INTEGRATION_TESTS=1"
)

pytestmark = pytest.mark.skipif(
    not (_CUDA_AVAILABLE and _INTEGRATION_ENABLED),
    reason=_SKIP_REASON,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ENCODER_MODEL_PATH = os.getenv(
    "ENCODER_MODEL_PATH", "jinaai/jina-embeddings-v4"
)

# 20 public-domain test images (COCO-style URLs — replace with real URLs).
_TEST_IMAGE_URLS: list[str] = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png",
] * 20  # Placeholder: same image 20× for shape/smoke testing


@pytest.fixture(scope="module")
def gpu_encoder():
    """Load a real FDE encoder onto GPU (module-scoped to amortise load time)."""
    from fde.models.jinavera import JinaVeRAFdeEncoder  # type: ignore[import]

    enc = JinaVeRAFdeEncoder(encoder_model_path=ENCODER_MODEL_PATH)
    enc = enc.cuda()
    yield enc
    enc.__finalize__()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImageRankerGpuSingleQuery:
    """GPU smoke tests for ImageRanker.rank — single query."""

    @pytest.mark.asyncio
    async def test_rank_returns_top_k(self, gpu_encoder: object) -> None:
        """rank() returns exactly k results on GPU."""
        from asmr.match import ImageRanker, RankerConfig

        ranker = ImageRanker(
            encoder=gpu_encoder,
            config=RankerConfig(top_k=5, device="cuda"),
        )
        results = await ranker.rank("a colorful abstract image", _TEST_IMAGE_URLS, k=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_rank_scores_are_finite(self, gpu_encoder: object) -> None:
        """All scores are finite floats."""
        from asmr.match import ImageRanker, RankerConfig

        ranker = ImageRanker(
            encoder=gpu_encoder,
            config=RankerConfig(top_k=3, device="cuda"),
        )
        results = await ranker.rank("transparent png", _TEST_IMAGE_URLS, k=3)
        assert all(np.isfinite(r.score) for r in results)

    @pytest.mark.asyncio
    async def test_rank_positions_sequential(self, gpu_encoder: object) -> None:
        """rank positions are 1, 2, 3, … with no gaps."""
        from asmr.match import ImageRanker, RankerConfig

        ranker = ImageRanker(
            encoder=gpu_encoder,
            config=RankerConfig(top_k=4, device="cuda"),
        )
        results = await ranker.rank("abstract", _TEST_IMAGE_URLS, k=4)
        assert [r.rank for r in results] == [1, 2, 3, 4]


class TestImageRankerGpuBatch:
    """GPU smoke tests for ImageRanker.rank_batch."""

    @pytest.mark.asyncio
    async def test_batch_returns_results_per_query(
        self, gpu_encoder: object
    ) -> None:
        """rank_batch returns one list per query."""
        from asmr.match import ImageRanker, RankerConfig

        ranker = ImageRanker(
            encoder=gpu_encoder,
            config=RankerConfig(top_k=3, device="cuda"),
        )
        queries = [
            ("colorful pattern", _TEST_IMAGE_URLS),
            ("dark background image", _TEST_IMAGE_URLS),
        ]
        results = await ranker.rank_batch(queries, k=3)
        assert len(results) == 2
        assert all(len(r) == 3 for r in results)


class TestImageRankerGpuWithAggregationHead:
    """GPU smoke tests with MFARFieldAdapter head."""

    @pytest.mark.asyncio
    async def test_rank_with_mfar_head(self, gpu_encoder: object) -> None:
        """rank() produces valid results when a trained head is applied."""
        from asmr.match import ImageRanker, RankerConfig
        from asmr.train.aggregation import MFARFieldAdapter

        head = MFARFieldAdapter(
            query_dim=64, num_fields=1, num_scorers=1
        ).cuda()
        ranker = ImageRanker(
            encoder=gpu_encoder,
            config=RankerConfig(top_k=5, device="cuda"),
        )
        results = await ranker.rank(
            "abstract art",
            _TEST_IMAGE_URLS,
            k=5,
            aggregation_head=head,
            query_encoder=gpu_encoder,
        )
        assert len(results) == 5
        assert results[0].rank == 1
