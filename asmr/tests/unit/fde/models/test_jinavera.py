import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch, call
from PIL import Image


def make_image() -> Image.Image:
    return Image.new("RGB", (10, 10))


@pytest.fixture
def mock_jinavera():
    """Return a Jinavera instance with model loading fully mocked out."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with (
        patch("fde.config.FdeConfig.apply_with_prompt_type", return_value=MagicMock()),
        patch("fde.models.jinavera.Jinavera._initialize_model") as mock_init_model,
        patch("fde.base.muvfde") as mock_muvfde,
    ):
        mock_muvfde.generate_fixed_dimensional_encoding.return_value = np.zeros(1600)

        from fde.models.jinavera import Jinavera

        encoder = Jinavera("model/path")
        # _initialize_model이 mock되었으므로 수동으로 설정
        encoder.encoder_model = mock_model
        encoder.tokenizer = mock_tokenizer

        yield encoder, mock_model, mock_muvfde


class TestJinaveraModelInitialization:
    @patch("fde.models.jinavera.jina_embeddings_v4")
    @patch("fde.models.jinavera.transformers")
    @patch("fde.base.muvfde")
    def test_initialize_model_loads_encoder_and_tokenizer(
        self, mock_muvfde, mock_transformers, mock_jina_module
    ):
        mock_model = MagicMock()
        mock_jina_module.JinaEmbeddingsV4Model.from_pretrained.return_value.to.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
        mock_muvfde.generate_fixed_dimensional_encoding.return_value = np.zeros(1600)

        from fde.models.jinavera import Jinavera
        from fde.config import PromptType, FdeConfig

        encoder = Jinavera.__new__(Jinavera)
        encoder.encoder_model_path = "model/path"
        encoder.fde_configs = {
            PromptType.QUERY: MagicMock(),
            PromptType.PASSAGE: MagicMock(),
        }
        encoder._initialize_model()

        mock_jina_module.JinaEmbeddingsV4Model.from_pretrained.assert_called_once_with(
            "model/path", trust_remote_code=False
        )
        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once()

    @patch("fde.models.jinavera.jina_embeddings_v4")
    @patch("fde.models.jinavera.transformers")
    @patch("fde.base.muvfde")
    def test_initialize_model_moves_to_cuda(
        self, mock_muvfde, mock_transformers, mock_jina_module
    ):
        mock_base_model = MagicMock()
        mock_jina_module.JinaEmbeddingsV4Model.from_pretrained.return_value = mock_base_model
        mock_muvfde.generate_fixed_dimensional_encoding.return_value = np.zeros(1600)

        from fde.models.jinavera import Jinavera

        encoder = Jinavera.__new__(Jinavera)
        encoder.encoder_model_path = "model/path"
        encoder.fde_configs = {}
        encoder._initialize_model()

        mock_base_model.to.assert_called_once_with("cuda")


class TestJinaveraTextEncoding:
    def test_encode_texts_to_multivector_calls_model(self, mock_jinavera):
        encoder, mock_model, _ = mock_jinavera
        mock_model.encode_text.return_value = [torch.randn(5, 8)]
        from fde.config import PromptType

        encoder._encode_texts_to_multivector(["hello world"], PromptType.QUERY)

        mock_model.encode_text.assert_called_once()
        _, kwargs = mock_model.encode_text.call_args
        assert kwargs.get("return_multivector") is True
        assert kwargs.get("prompt_name") == "query"

    def test_encode_texts_uses_retrieval_task(self, mock_jinavera):
        encoder, mock_model, _ = mock_jinavera
        mock_model.encode_text.return_value = [torch.randn(5, 8)]
        from fde.config import PromptType

        encoder._encode_texts_to_multivector(["text"], PromptType.PASSAGE)

        _, kwargs = mock_model.encode_text.call_args
        assert kwargs.get("task") == "retrieval"

    def test_encode_text_returns_numpy_array(self, mock_jinavera):
        encoder, mock_model, mock_muvfde = mock_jinavera
        mock_model.encode_text.return_value = [torch.randn(5, 8)]
        mock_muvfde.generate_fixed_dimensional_encoding.return_value = np.zeros(1600)
        from fde.config import PromptType

        result = encoder.encode_text("hello", PromptType.QUERY)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1


class TestJinaveraImageEncoding:
    def test_encode_images_to_multivector_calls_model(self, mock_jinavera):
        encoder, mock_model, _ = mock_jinavera
        mock_model.encode_image.return_value = [torch.randn(5, 8)]

        encoder._encode_images_to_multivector([make_image()])

        mock_model.encode_image.assert_called_once()
        _, kwargs = mock_model.encode_image.call_args
        assert kwargs.get("return_multivector") is True

    def test_encode_images_uses_retrieval_task(self, mock_jinavera):
        encoder, mock_model, _ = mock_jinavera
        mock_model.encode_image.return_value = [torch.randn(5, 8)]

        encoder._encode_images_to_multivector([make_image()])

        _, kwargs = mock_model.encode_image.call_args
        assert kwargs.get("task") == "retrieval"

    def test_encode_image_returns_numpy_array(self, mock_jinavera):
        encoder, mock_model, mock_muvfde = mock_jinavera
        mock_model.encode_image.return_value = [torch.randn(5, 8)]
        mock_muvfde.generate_fixed_dimensional_encoding.return_value = np.zeros(1600)

        result = encoder.encode_image(make_image())

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1


class TestFdeEncoderCleanup:
    def test_finalize_deletes_model(self, mock_jinavera):
        encoder, mock_model, _ = mock_jinavera

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            encoder.__finalize__()

        assert not hasattr(encoder, "encoder_model")
        mock_empty_cache.assert_called_once()

    def test_finalize_deletes_tokenizer_if_present(self, mock_jinavera):
        encoder, _, _ = mock_jinavera

        with patch("torch.cuda.empty_cache"):
            encoder.__finalize__()

        assert not hasattr(encoder, "tokenizer")

    def test_finalize_skips_tokenizer_if_absent(self, mock_jinavera):
        encoder, _, _ = mock_jinavera
        del encoder.tokenizer  # remove before finalize

        with patch("torch.cuda.empty_cache"):
            encoder.__finalize__()  # should not raise
