
from typing import Any


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

class TokenizerWrapper:

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)
