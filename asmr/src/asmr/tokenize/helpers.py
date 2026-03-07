import io
from typing import Any, Union
from PIL import Image
from transformers import AutoTokenizer
from kiwipiepy import Kiwi

from asmr.tunip import preprocess
from asmr.index import config


def split_text(text: str, max_length: int) -> list[str]:
    """Splits the input text into chunks of at most max_length characters."""
    words = text.split()
    chunks = []
    current_chunk: list[str] = []

    for word in words:
        # If adding the next word exceeds max_length, finalize the current chunk
        if current_chunk and len(" ".join(current_chunk + [word])) > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(word)

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class SplitTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


class AutoTokenizerWrapper:
    """Wrapper for HuggingFace AutoTokenizer"""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)


class TokenizerWrapper:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    @classmethod
    def from_auto_tokenizer(cls, model_path: str) -> "TokenizerWrapper":
        """Create TokenizerWrapper from HuggingFace AutoTokenizer"""
        auto_tokenizer = AutoTokenizerWrapper(model_path)
        return cls(auto_tokenizer)

    @classmethod
    def from_morph_tokenizer(cls) -> "TokenizerWrapper":
        """Create TokenizerWrapper from morphological tokenizer"""
        morph_tokenizer = MorphTokenizerWrapper()
        return cls(morph_tokenizer)

    @classmethod
    def from_split_tokenizer(cls) -> "TokenizerWrapper":
        """Create TokenizerWrapper from SplitTokenizer"""
        split_tokenizer = SplitTokenizer()
        return cls(split_tokenizer)

    @classmethod
    def from_config(cls, field_config: config.FieldConfig) -> "TokenizerWrapper":
        """Create TokenizerWrapper from FieldConfig

        Args:
            field_config: FieldConfig containing tokenizer_type and model_path

        Returns:
            TokenizerWrapper instance configured according to field_config

        Raises:
            ValueError: If tokenizer_type is not supported
        """
        if field_config.tokenizer_type == config.TokenizerType.MORPH:
            return cls.from_morph_tokenizer()
        elif field_config.tokenizer_type == config.TokenizerType.SPLIT:
            return cls.from_split_tokenizer()
        elif field_config.tokenizer_type == config.TokenizerType.HF_AUTO:
            return cls.from_auto_tokenizer(field_config.model_path)
        else:
            raise ValueError(
                f"Unsupported tokenizer type: {field_config.tokenizer_type}"
            )


class MorphTokenizerWrapper:
    """Wrapper for Kiwi morphological tokenizer"""

    def __init__(self):
        if Kiwi is None:
            raise ImportError(
                "kiwipiepy is not installed. Please install it with: pip install kiwipiepy"
            )
        self.kiwi = Kiwi()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using Kiwi morphological analyzer

        Args:
            text: Input text to tokenize

        Returns:
            List of token forms (surface forms)
        """
        text = preprocess.preprocess_korean(text, strict=False)
        tokens = self.kiwi.tokenize(text)
        return [token.form.lower() for token in tokens]

    def tokenize_with_tags(self, text: str) -> list[dict]:
        """Tokenize text and return detailed token information

        Args:
            text: Input text to tokenize

        Returns:
            List of dictionaries containing token information (form, tag, start, len)
        """
        tokens = self.kiwi.tokenize(text)
        return [
            {
                "form": token.form,
                "tag": token.tag,
                "start": token.start,
                "len": token.len,
            }
            for token in tokens
        ]


def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """Convert image bytes to PIL Image"""
    return Image.open(io.BytesIO(image_bytes))


class ChunkProcessor:
    """Representative class to handle chunking of text or images"""

    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str, tokenizer_wrapper: TokenizerWrapper) -> list[str]:
        """Chunk text into smaller pieces"""
        return split_text(text, self.max_chunk_size)

    def chunk_image(
        self, image: Union[Image.Image, bytes], tile_size: tuple[int, int] = (224, 224)
    ) -> list[Image.Image]:
        """Chunk image into tiles"""
        if isinstance(image, bytes):
            image = bytes_to_pil_image(image)

        width, height = image.size
        tile_width, tile_height = tile_size
        chunks = []

        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                box = (x, y, min(x + tile_width, width), min(y + tile_height, height))
                chunk = image.crop(box)
                chunks.append(chunk)

        return chunks
