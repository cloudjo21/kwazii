
import io
from typing import Any, Union
from PIL import Image
from transformers import AutoTokenizer


def split_text(text: str, max_length: int) -> list[str]:
    """Splits the input text into chunks of at most max_length characters."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        # If adding the next word exceeds max_length, finalize the current chunk
        if current_chunk and len(' '.join(current_chunk + [word])) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
        current_chunk.append(word)

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

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
    def from_auto_tokenizer(cls, model_path: str) -> 'TokenizerWrapper':
        """Create TokenizerWrapper from HuggingFace AutoTokenizer"""
        auto_tokenizer = AutoTokenizerWrapper(model_path)
        return cls(auto_tokenizer)
    
    @classmethod
    def from_split_tokenizer(cls) -> 'TokenizerWrapper':
        """Create TokenizerWrapper from SplitTokenizer"""
        split_tokenizer = SplitTokenizer()
        return cls(split_tokenizer)


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
    
    def chunk_image(self, image: Union[Image.Image, bytes], tile_size: tuple[int, int] = (224, 224)) -> list[Image.Image]:
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
