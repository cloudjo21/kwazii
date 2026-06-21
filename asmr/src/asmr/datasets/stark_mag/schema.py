"""STaRK-MAG dataset schema implementing DatasetSchema."""

from pathlib import Path
from typing import Any

from asmr.datasets.stark_mag.loader import (
    MAG_FIELD_NAMES,
    MagCorpus,
    MagQuery,
    build_mag_corpus,
    field_text,
    load_mag_queries,
)


class StarkMagSchema:
    """DatasetSchema implementation for the STaRK-MAG academic paper benchmark."""

    name: str = "stark_mag"
    field_names: tuple[str, ...] = MAG_FIELD_NAMES

    def load_queries(self, data_root: Path, split: str) -> list[MagQuery]:
        return load_mag_queries(data_root, split)

    def build_corpus(self, data_root: Path, *, max_docs: int = -1) -> MagCorpus:
        return build_mag_corpus(data_root, max_docs)

    def field_text(self, document: dict[str, Any], field_name: str) -> str:
        return field_text(document, field_name)


STARK_MAG_SCHEMA = StarkMagSchema()
