import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image


def make_image() -> Image.Image:
    return Image.new("RGB", (10, 10))


class ConcreteEncoder:
    """Minimal concrete implementation of BaseFdeEncoder for testing."""

    def __init__(self, encoder_model_path: str, multivector_outputs: list | None = None):
        from fde.config import PromptType, FdeConfig
        self.encoder_model_path = encoder_model_path
        self.fde_configs = {
            PromptType.QUERY: MagicMock(),
            PromptType.PASSAGE: MagicMock(),
        }
        self._multivector_outputs = multivector_outputs or []
        self._finalized = False

    def _initialize_model(self, **kwargs):
        pass

    def _encode_texts_to_multivector(self, texts, prompt_type):
        import torch
        return [torch.randn(5, 8) for _ in texts]

    def _encode_images_to_multivector(self, images):
        import torch
        return [torch.randn(5, 8) for _ in images]

    def __finalize__(self):
        self._finalized = True


class TestBaseFdeEncoderInterface:
    def test_base_encoder_abstract_methods_enforced(self):
        """BaseFdeEncoder cannot be instantiated without implementing abstract methods."""
        from fde.base import BaseFdeEncoder

        with pytest.raises(TypeError):
            BaseFdeEncoder("some/path")

    def test_concrete_subclass_can_be_instantiated(self):
        encoder = ConcreteEncoder("model/path")

        assert encoder.encoder_model_path == "model/path"

    def test_fde_configs_has_query_and_passage(self):
        from fde.config import PromptType

        encoder = ConcreteEncoder("model/path")

        assert PromptType.QUERY in encoder.fde_configs
        assert PromptType.PASSAGE in encoder.fde_configs

    def test_finalize_is_callable(self):
        encoder = ConcreteEncoder("model/path")

        encoder.__finalize__()

        assert encoder._finalized is True


class TestBaseFdeEncoderEncodeText:
    @patch("fde.base.muvfde")
    def test_encode_text_single_string(self, mock_muvfde):
        import torch
        from fde.base import BaseFdeEncoder
        from fde.config import PromptType

        mock_muvfde.generate_fixed_dimensional_encoding.return_value = np.zeros(1600)

        class _Encoder(BaseFdeEncoder):
            def _initialize_model(self, **kwargs): pass
            def _encode_texts_to_multivector(self, texts, prompt_type):
                return [torch.randn(5, 8) for _ in texts]
            def _encode_images_to_multivector(self, images):
                return [torch.randn(5, 8) for _ in images]
            def __finalize__(self): pass

        encoder = _Encoder.__new__(_Encoder)
        encoder.encoder_model_path = "model/path"
        encoder.fde_configs = {
            PromptType.QUERY: MagicMock(),
            PromptType.PASSAGE: MagicMock(),
        }

        result = encoder.encode_text("hello world", PromptType.QUERY)

        assert result.shape[0] == 1  # batch size 1

    @patch("fde.base.muvfde")
    def test_encode_text_list_of_strings(self, mock_muvfde):
        import torch
        from fde.base import BaseFdeEncoder
        from fde.config import PromptType

        mock_muvfde.generate_fixed_dimensional_encoding.return_value = np.zeros(1600)

        class _Encoder(BaseFdeEncoder):
            def _initialize_model(self, **kwargs): pass
            def _encode_texts_to_multivector(self, texts, prompt_type):
                return [torch.randn(5, 8) for _ in texts]
            def _encode_images_to_multivector(self, images):
                return [torch.randn(5, 8) for _ in images]
            def __finalize__(self): pass

        encoder = _Encoder.__new__(_Encoder)
        encoder.encoder_model_path = "model/path"
        encoder.fde_configs = {
            PromptType.QUERY: MagicMock(),
            PromptType.PASSAGE: MagicMock(),
        }

        result = encoder.encode_text(["hello", "world"], PromptType.QUERY)

        assert result.shape[0] == 2  # batch size 2

    @patch("fde.base.muvfde")
    def test_encode_image_single_image(self, mock_muvfde):
        import torch
        from fde.base import BaseFdeEncoder
        from fde.config import PromptType

        mock_muvfde.generate_fixed_dimensional_encoding.return_value = np.zeros(1600)

        class _Encoder(BaseFdeEncoder):
            def _initialize_model(self, **kwargs): pass
            def _encode_texts_to_multivector(self, texts, prompt_type):
                return [torch.randn(5, 8) for _ in texts]
            def _encode_images_to_multivector(self, images):
                return [torch.randn(5, 8) for _ in images]
            def __finalize__(self): pass

        encoder = _Encoder.__new__(_Encoder)
        encoder.encoder_model_path = "model/path"
        encoder.fde_configs = {
            PromptType.QUERY: MagicMock(),
            PromptType.PASSAGE: MagicMock(),
        }

        result = encoder.encode_image(make_image())

        assert result.shape[0] == 1

    @patch("fde.base.muvfde")
    def test_encode_image_list_of_images(self, mock_muvfde):
        import torch
        from fde.base import BaseFdeEncoder
        from fde.config import PromptType

        mock_muvfde.generate_fixed_dimensional_encoding.return_value = np.zeros(1600)

        class _Encoder(BaseFdeEncoder):
            def _initialize_model(self, **kwargs): pass
            def _encode_texts_to_multivector(self, texts, prompt_type):
                return [torch.randn(5, 8) for _ in texts]
            def _encode_images_to_multivector(self, images):
                return [torch.randn(5, 8) for _ in images]
            def __finalize__(self): pass

        encoder = _Encoder.__new__(_Encoder)
        encoder.encoder_model_path = "model/path"
        encoder.fde_configs = {
            PromptType.QUERY: MagicMock(),
            PromptType.PASSAGE: MagicMock(),
        }

        result = encoder.encode_image([make_image(), make_image()])

        assert result.shape[0] == 2

    @patch("fde.base.muvfde")
    def test_similarity_returns_correct_shape(self, mock_muvfde):
        from fde.base import BaseFdeEncoder
        from fde.config import PromptType

        class _Encoder(BaseFdeEncoder):
            def _initialize_model(self, **kwargs): pass
            def _encode_texts_to_multivector(self, texts, prompt_type): return []
            def _encode_images_to_multivector(self, images): return []
            def __finalize__(self): pass

        encoder = _Encoder.__new__(_Encoder)
        encoder.fde_configs = {}

        a = np.random.rand(3, 16).astype(np.float32)
        b = np.random.rand(5, 16).astype(np.float32)

        result = encoder.similarity(a, b)

        assert result.shape == (3, 5)
