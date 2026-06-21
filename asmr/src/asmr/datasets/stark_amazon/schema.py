"""STaRK-Amazon dataset schema implementing DatasetSchema."""

from pathlib import Path
from typing import Any

from asmr.datasets.stark_amazon.loader import (
    AMAZON_FIELD_NAMES,
    AmazonCorpus,
    AmazonQuery,
    build_amazon_corpus,
    field_text,
    load_amazon_queries,
)


class StarkAmazonSchema:
    """DatasetSchema implementation for the STaRK-Amazon product benchmark."""

    name: str = "stark_amazon"
    field_names: tuple[str, ...] = AMAZON_FIELD_NAMES

    def load_queries(self, data_root: Path, split: str) -> list[AmazonQuery]:
        return load_amazon_queries(data_root, split)

    def build_corpus(self, data_root: Path, *, max_docs: int = -1) -> AmazonCorpus:
        return build_amazon_corpus(data_root, max_docs)

    def field_text(self, document: dict[str, Any], field_name: str) -> str:
        return field_text(document, field_name)


STARK_AMAZON_SCHEMA = StarkAmazonSchema()
