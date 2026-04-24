"""Unit tests for asmr.match.ranker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from asmr.match.config import RankerConfig
from asmr.match.ranker import ImageRanker, RankResult, _aggregate_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores_2d(n_fields: int = 1, n_docs: int = 5) -> np.ndarray:
    """Return a [F, D] score array."""
    rng = np.random.default_rng(42)
    return rng.random((n_fields, n_docs), dtype=np.float32)


def _make_doc_ids(n: int) -> list[str]:
    return [f"img_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# _aggregate_scores
# ---------------------------------------------------------------------------


class TestAggregateScores:
    """Tests for the internal _aggregate_scores utility."""

    def test_2d_returns_max_over_fields(self) -> None:
        """[F, D] input → max over F axis."""
        scores = np.array([[0.1, 0.5, 0.3], [0.4, 0.2, 0.6]], dtype=np.float32)
        result = _aggregate_scores(scores)
        np.testing.assert_array_almost_equal(
            result, np.array([0.4, 0.5, 0.6], dtype=np.float32)
        )

    def test_3d_returns_max_over_fields_and_scorers(self) -> None:
        """[F, M, D] input → max over F and M axes."""
        scores = np.zeros((2, 2, 3), dtype=np.float32)
        scores[0, 1, 2] = 0.9
        scores[1, 0, 0] = 0.7
        result = _aggregate_scores(scores)
        expected = np.array([0.7, 0.0, 0.9], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_1d_returns_flattened(self) -> None:
        """1-D input is returned as-is (flattened)."""
        scores = np.array([0.3, 0.1, 0.8], dtype=np.float32)
        result = _aggregate_scores(scores)
        np.testing.assert_array_equal(result, scores)


# ---------------------------------------------------------------------------
# ImageRanker
# ---------------------------------------------------------------------------


@pytest.fixture()
def ranker(mock_encoder: MagicMock) -> ImageRanker:
    """ImageRanker with default config and mock encoder."""
    return ImageRanker(encoder=mock_encoder, config=RankerConfig(top_k=5))


@pytest.fixture()
def mock_retriever() -> MagicMock:
    """Mock DocumentRetriever that returns sorted doc_ids and [1, N] scores."""
    r = MagicMock()
    return r


class TestImageRankerInit:
    """Tests for ImageRanker construction."""

    def test_default_config_used_when_none(self, mock_encoder: MagicMock) -> None:
        """When config=None a default RankerConfig is created."""
        ranker = ImageRanker(encoder=mock_encoder)
        assert ranker._config == RankerConfig()

    def test_injected_config_stored(self, mock_encoder: MagicMock) -> None:
        """Custom config is stored on the instance."""
        cfg = RankerConfig(top_k=3, device="cuda")
        ranker = ImageRanker(encoder=mock_encoder, config=cfg)
        assert ranker._config is cfg


class TestImageRankerRankValidation:
    """Tests for input validation in ImageRanker.rank."""

    @pytest.mark.asyncio
    async def test_empty_text_raises(self, ranker: ImageRanker) -> None:
        """Empty text raises ValueError."""
        with pytest.raises(ValueError, match="text must be non-empty"):
            await ranker.rank("", ["http://a.com/img.jpg"])

    @pytest.mark.asyncio
    async def test_empty_urls_raises(self, ranker: ImageRanker) -> None:
        """Empty image_urls list raises ValueError."""
        with pytest.raises(ValueError, match="image_urls must be non-empty"):
            await ranker.rank("shirt", [])


class TestImageRankerRankNoHead:
    """Tests for ImageRanker.rank without aggregation head."""

    @pytest.mark.asyncio
    async def test_returns_top_k_results(
        self, ranker: ImageRanker, image_urls: list[str]
    ) -> None:
        """Returns exactly k results for k < len(image_urls)."""
        n = len(image_urls)  # 20
        scores = np.linspace(0.1, 1.0, n, dtype=np.float32).reshape(1, n)
        doc_ids = _make_doc_ids(n)

        mock_ret = AsyncMock(return_value=(doc_ids, scores))
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_aggregation_head = mock_ret

        with (
            patch.object(
                ranker._indexer,
                "build",
                return_value=(mock_retriever, doc_ids),
            ),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank("shirt", image_urls, k=5)

        assert len(results) == 5
        assert all(isinstance(r, RankResult) for r in results)

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_descending(
        self, ranker: ImageRanker, image_urls: list[str]
    ) -> None:
        """Results are ordered from highest to lowest score."""
        n = 5
        urls = image_urls[:n]
        scores = np.array([[0.3, 0.9, 0.1, 0.7, 0.5]], dtype=np.float32)
        doc_ids = _make_doc_ids(n)

        mock_ret = AsyncMock(return_value=(doc_ids, scores))
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_aggregation_head = mock_ret

        with (
            patch.object(
                ranker._indexer,
                "build",
                return_value=(mock_retriever, doc_ids),
            ),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank("shirt", urls, k=3)

        assert results[0].score >= results[1].score >= results[2].score

    @pytest.mark.asyncio
    async def test_k_larger_than_urls_clamped(self, ranker: ImageRanker) -> None:
        """k > len(image_urls) is silently clamped to len(image_urls)."""
        urls = ["http://a.com/img.jpg"] * 3
        scores = np.ones((1, 3), dtype=np.float32)
        doc_ids = _make_doc_ids(3)

        mock_ret = AsyncMock(return_value=(doc_ids, scores))
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_aggregation_head = mock_ret

        with (
            patch.object(
                ranker._indexer, "build", return_value=(mock_retriever, doc_ids)
            ),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank("shirt", urls, k=10)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_rank_positions_are_sequential(
        self, ranker: ImageRanker, image_urls: list[str]
    ) -> None:
        """rank field is 1-based and consecutive."""
        n = 5
        urls = image_urls[:n]
        scores = np.arange(n, dtype=np.float32).reshape(1, n)
        doc_ids = _make_doc_ids(n)

        mock_ret = AsyncMock(return_value=(doc_ids, scores))
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_aggregation_head = mock_ret

        with (
            patch.object(
                ranker._indexer, "build", return_value=(mock_retriever, doc_ids)
            ),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank("shirt", urls, k=n)

        assert [r.rank for r in results] == list(range(1, n + 1))

    @pytest.mark.asyncio
    async def test_urls_mapped_correctly_from_doc_ids(
        self, ranker: ImageRanker
    ) -> None:
        """RankResult.url corresponds to the original input URL."""
        urls = ["http://a.com/cat.jpg", "http://a.com/dog.jpg"]
        doc_ids = ["img_0", "img_1"]
        scores = np.array([[0.9, 0.1]], dtype=np.float32)

        mock_ret = AsyncMock(return_value=(doc_ids, scores))
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_aggregation_head = mock_ret

        with (
            patch.object(
                ranker._indexer, "build", return_value=(mock_retriever, doc_ids)
            ),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank("animal", urls, k=2)

        assert results[0].url == "http://a.com/cat.jpg"
        assert results[1].url == "http://a.com/dog.jpg"

    @pytest.mark.asyncio
    async def test_3d_scores_aggregated_correctly(self, ranker: ImageRanker) -> None:
        """[F, M, D] scores are max-aggregated before ranking."""
        urls = ["u0", "u1", "u2"]
        doc_ids = ["img_0", "img_1", "img_2"]
        scores = np.zeros((1, 2, 3), dtype=np.float32)
        scores[0, 1, 1] = 0.99  # img_1 should rank first

        mock_ret = AsyncMock(return_value=(doc_ids, scores))
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_aggregation_head = mock_ret

        with (
            patch.object(
                ranker._indexer, "build", return_value=(mock_retriever, doc_ids)
            ),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank("item", urls, k=1)

        assert results[0].url == "u1"


class TestImageRankerRankWithHead:
    """Tests for ImageRanker.rank with an aggregation head."""

    @pytest.mark.asyncio
    async def test_uses_head_sorted_order(
        self, ranker: ImageRanker, image_urls: list[str]
    ) -> None:
        """When head is provided, returned_ids order is used directly."""
        n = 5
        urls = image_urls[:n]
        # Simulate apply_aggregation_head returning sorted order (img_4 first)
        sorted_ids = ["img_4", "img_2", "img_0", "img_3", "img_1"]
        logits = np.array([0.95, 0.80, 0.70, 0.60, 0.50], dtype=np.float32)

        mock_ret = AsyncMock(return_value=(sorted_ids, logits))
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_aggregation_head = mock_ret

        mock_head = MagicMock()
        doc_ids = _make_doc_ids(n)

        with (
            patch.object(
                ranker._indexer, "build", return_value=(mock_retriever, doc_ids)
            ),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank("shirt", urls, k=3, aggregation_head=mock_head)

        assert len(results) == 3
        assert results[0].url == urls[4]
        assert results[1].url == urls[2]

    @pytest.mark.asyncio
    async def test_head_scores_reflect_logits(
        self, ranker: ImageRanker, image_urls: list[str]
    ) -> None:
        """RankResult.score reflects the logit values from the head."""
        n = 3
        urls = image_urls[:n]
        sorted_ids = ["img_0", "img_1", "img_2"]
        logits = np.array([0.9, 0.6, 0.3], dtype=np.float32)

        mock_ret = AsyncMock(return_value=(sorted_ids, logits))
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_aggregation_head = mock_ret

        doc_ids = _make_doc_ids(n)
        with (
            patch.object(
                ranker._indexer, "build", return_value=(mock_retriever, doc_ids)
            ),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank(
                "shirt", urls, k=3, aggregation_head=MagicMock()
            )

        assert pytest.approx(results[0].score, abs=1e-5) == 0.9
        assert pytest.approx(results[1].score, abs=1e-5) == 0.6


class TestImageRankerBatch:
    """Tests for ImageRanker.rank_batch."""

    @pytest.mark.asyncio
    async def test_empty_queries_returns_empty(self, ranker: ImageRanker) -> None:
        """rank_batch([]) returns an empty list."""
        result = await ranker.rank_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_length_matches_input(
        self, ranker: ImageRanker, image_urls: list[str]
    ) -> None:
        """rank_batch returns one result list per query."""
        urls_a = image_urls[:5]
        urls_b = image_urls[5:10]
        queries = [("shirt", urls_a), ("pants", urls_b)]

        n = 5
        scores = np.ones((1, n), dtype=np.float32)
        doc_ids_a = _make_doc_ids(n)
        doc_ids_b = _make_doc_ids(n)

        mock_ret_a = AsyncMock(return_value=(doc_ids_a, scores))
        mock_ret_b = AsyncMock(return_value=(doc_ids_b, scores))

        mock_retriever_a = MagicMock()
        mock_retriever_a.retrieve_with_aggregation_head = mock_ret_a
        mock_retriever_b = MagicMock()
        mock_retriever_b.retrieve_with_aggregation_head = mock_ret_b

        build_returns = [
            (mock_retriever_a, doc_ids_a),
            (mock_retriever_b, doc_ids_b),
        ]

        with (
            patch.object(ranker._indexer, "build", side_effect=build_returns),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank_batch(queries, k=3)

        assert len(results) == 2
        assert len(results[0]) == 3
        assert len(results[1]) == 3

    @pytest.mark.asyncio
    async def test_batch_each_query_gets_correct_urls(
        self, ranker: ImageRanker
    ) -> None:
        """Each query in the batch is ranked against its own URL set."""
        urls_a = ["http://a.com/0.jpg", "http://a.com/1.jpg"]
        urls_b = ["http://b.com/0.jpg", "http://b.com/1.jpg"]
        queries = [("query_a", urls_a), ("query_b", urls_b)]

        scores = np.array([[0.8, 0.2]], dtype=np.float32)

        def _make_mock_ret(doc_ids: list[str]) -> AsyncMock:
            return AsyncMock(return_value=(doc_ids, scores))

        mock_retrievers = []
        doc_ids_list = []
        for urls in [urls_a, urls_b]:
            doc_ids = [f"img_{i}" for i in range(len(urls))]
            doc_ids_list.append(doc_ids)
            mr = MagicMock()
            mr.retrieve_with_aggregation_head = _make_mock_ret(doc_ids)
            mock_retrievers.append(mr)

        build_returns = list(zip(mock_retrievers, doc_ids_list))
        with (
            patch.object(ranker._indexer, "build", side_effect=build_returns),
            patch("asmr.retrieve.query.Query"),
        ):
            results = await ranker.rank_batch(queries, k=1)

        # First result should map to urls_a, second to urls_b
        assert results[0][0].url.startswith("http://a.com/")
        assert results[1][0].url.startswith("http://b.com/")
