"""Unit tests for asmr.match.indexer."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from asmr.match.config import RankerConfig
from asmr.match.indexer import CandidateImageIndexer, _load_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_image() -> Image.Image:
    """Create a minimal 4x4 RGB PIL image for tests."""
    return Image.new("RGB", (4, 4), color=(128, 64, 32))


def _make_fake_jpeg_bytes() -> bytes:
    """Encode a small PIL image as JPEG bytes."""
    buf = io.BytesIO()
    _make_fake_image().save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    """Tests for the module-level _load_image helper."""

    def test_local_path_success(self, tmp_path: pytest.TempdirFactory) -> None:
        """Loads a local image file by path."""
        img_path = tmp_path / "test.jpg"
        _make_fake_image().save(str(img_path))
        result = _load_image(str(img_path), timeout_s=5.0)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    @patch("urllib.request.urlopen")
    def test_http_url_success(self, mock_urlopen: MagicMock) -> None:
        """Loads an image from an HTTP URL."""
        resp_mock = MagicMock()
        resp_mock.read.return_value = _make_fake_jpeg_bytes()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp_mock

        result = _load_image("http://example.com/img.jpg", timeout_s=5.0)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen", side_effect=OSError("connection refused"))
    def test_http_url_failure_raises_value_error(
        self, _mock_urlopen: MagicMock
    ) -> None:
        """Raises ValueError when URL cannot be reached."""
        with pytest.raises(ValueError, match="Failed to load image"):
            _load_image("http://bad-host/img.jpg", timeout_s=1.0)

    def test_missing_local_path_raises_value_error(
        self, tmp_path: pytest.TempdirFactory
    ) -> None:
        """Raises ValueError for a non-existent local path."""
        with pytest.raises(ValueError, match="Failed to load image"):
            _load_image(str(tmp_path / "nonexistent.jpg"), timeout_s=5.0)

    @pytest.mark.parametrize(
        "url",
        ["http://x.com/a.jpg", "https://x.com/b.png"],
    )
    @patch("urllib.request.urlopen")
    def test_https_url_also_works(self, mock_urlopen: MagicMock, url: str) -> None:
        """Both http:// and https:// schemes are treated as URLs."""
        resp_mock = MagicMock()
        resp_mock.read.return_value = _make_fake_jpeg_bytes()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp_mock

        result = _load_image(url, timeout_s=5.0)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# CandidateImageIndexer
# ---------------------------------------------------------------------------


class TestCandidateImageIndexerLoadImages:
    """Tests for CandidateImageIndexer.load_images."""

    def test_returns_pil_images(self, mock_encoder: MagicMock) -> None:
        """load_images returns one PIL Image per URL."""
        config = RankerConfig()
        indexer = CandidateImageIndexer(encoder=mock_encoder, config=config)

        fake_img = _make_fake_image()
        with patch(
            "asmr.match.indexer._load_image", return_value=fake_img
        ) as mock_load:
            urls = ["http://a.com/1.jpg", "http://a.com/2.jpg"]
            result = indexer.load_images(urls)

        assert len(result) == 2
        assert all(isinstance(img, Image.Image) for img in result)
        assert mock_load.call_count == 2

    def test_propagates_load_error(self, mock_encoder: MagicMock) -> None:
        """load_images propagates ValueError from _load_image."""
        config = RankerConfig()
        indexer = CandidateImageIndexer(encoder=mock_encoder, config=config)

        with patch(
            "asmr.match.indexer._load_image",
            side_effect=ValueError("bad URL"),
        ):
            with pytest.raises(ValueError, match="bad URL"):
                indexer.load_images(["http://bad.com/img.jpg"])

    def test_passes_timeout_to_load(self, mock_encoder: MagicMock) -> None:
        """load_images passes config.image_load_timeout_s to _load_image."""
        config = RankerConfig(image_load_timeout_s=3.0)
        indexer = CandidateImageIndexer(encoder=mock_encoder, config=config)

        fake_img = _make_fake_image()
        with patch(
            "asmr.match.indexer._load_image", return_value=fake_img
        ) as mock_load:
            indexer.load_images(["http://a.com/img.jpg"])

        assert mock_load.call_args[0][1] == 3.0


class TestCandidateImageIndexerBuild:
    """Tests for CandidateImageIndexer.build."""

    def test_empty_urls_raises_value_error(self, mock_encoder: MagicMock) -> None:
        """build() raises ValueError when image_urls is empty."""
        indexer = CandidateImageIndexer(encoder=mock_encoder, config=RankerConfig())
        with pytest.raises(ValueError, match="non-empty"):
            indexer.build([])

    def test_returns_retriever_and_doc_ids(self, mock_encoder: MagicMock) -> None:
        """build() returns (DocumentRetriever, doc_ids) with correct length."""
        indexer = CandidateImageIndexer(encoder=mock_encoder, config=RankerConfig())
        urls = [f"http://x.com/{i}.jpg" for i in range(5)]

        mock_retriever = MagicMock()
        mock_doc_ids = [f"img_{i}" for i in range(5)]

        with (
            patch.object(indexer, "load_images", return_value=[_make_fake_image()] * 5),
            patch("asmr.index.fields.DenseImageFieldIndex") as MockFieldIndex,
            patch("asmr.retrieve.retrievers.DenseImageFieldRetriever") as MockRetriever,
            patch("asmr.retrieve.retrievers.QueryRouter") as MockRouter,
            patch(
                "asmr.retrieve.helpers.DocumentRetriever", return_value=mock_retriever
            ),
        ):
            MockFieldIndex.return_value = MagicMock()
            MockRetriever.return_value = MagicMock()
            MockRouter.return_value = MagicMock()

            retriever, doc_ids = indexer.build(urls)

        assert retriever is mock_retriever
        assert len(doc_ids) == 5
        assert doc_ids == mock_doc_ids

    def test_doc_ids_follow_img_prefix_pattern(self, mock_encoder: MagicMock) -> None:
        """doc_ids are 'img_0', 'img_1', ... matching index positions."""
        indexer = CandidateImageIndexer(encoder=mock_encoder, config=RankerConfig())
        urls = ["http://a.com/x.jpg"] * 3

        with (
            patch.object(indexer, "load_images", return_value=[_make_fake_image()] * 3),
            patch("asmr.index.fields.DenseImageFieldIndex"),
            patch("asmr.retrieve.retrievers.DenseImageFieldRetriever"),
            patch("asmr.retrieve.retrievers.QueryRouter"),
            patch("asmr.retrieve.helpers.DocumentRetriever"),
        ):
            _, doc_ids = indexer.build(urls)

        assert doc_ids == ["img_0", "img_1", "img_2"]

    def test_build_uses_configured_field_name(self, mock_encoder: MagicMock) -> None:
        """build() passes config.field_name to QueryRouter."""
        config = RankerConfig(field_name="thumb")
        indexer = CandidateImageIndexer(encoder=mock_encoder, config=config)
        urls = ["http://a.com/img.jpg"]

        with (
            patch.object(indexer, "load_images", return_value=[_make_fake_image()]),
            patch("asmr.index.fields.DenseImageFieldIndex"),
            patch("asmr.retrieve.retrievers.DenseImageFieldRetriever"),
            patch("asmr.retrieve.retrievers.QueryRouter") as MockRouter,
            patch("asmr.retrieve.helpers.DocumentRetriever"),
        ):
            indexer.build(urls)

        call_kwargs = MockRouter.call_args[0][0]
        assert "thumb" in call_kwargs
