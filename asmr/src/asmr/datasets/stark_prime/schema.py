"""STaRK-Prime dataset schema implementing DatasetSchema."""

from pathlib import Path
from typing import Any

from asmr.datasets.stark_prime.loader import (
    PRIME_FIELD_NAMES,
    PrimeCorpus,
    PrimeQuery,
    build_prime_corpus,
    field_text,
    load_prime_queries,
)


class StarkPrimeSchema:
    """DatasetSchema implementation for the STaRK-Prime biomedical benchmark."""

    name: str = "stark_prime"
    field_names: tuple[str, ...] = PRIME_FIELD_NAMES

    def load_queries(self, data_root: Path, split: str) -> list[PrimeQuery]:
        return load_prime_queries(data_root, split)

    def build_corpus(self, data_root: Path, *, max_docs: int = -1) -> PrimeCorpus:
        return build_prime_corpus(data_root, max_docs)

    def field_text(self, document: dict[str, Any], field_name: str) -> str:
        return field_text(document, field_name)


STARK_PRIME_SCHEMA = StarkPrimeSchema()
