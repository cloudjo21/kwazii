"""Dataset schema registry for multi-dataset benchmark support."""

from typing import Any, Protocol, runtime_checkable
from pathlib import Path


@runtime_checkable
class DatasetSchema(Protocol):
    """Per-dataset schema: field list + data loading contract.

    Implement this protocol for each new dataset (stark_mag, stark_amazon, etc.).
    Register the instance with register_dataset() so benchmark infrastructure
    can resolve it by name via get_dataset().
    """

    @property
    def name(self) -> str:
        """Unique dataset identifier (e.g. 'stark_prime', 'stark_mag')."""
        ...

    @property
    def field_names(self) -> tuple[str, ...]:
        """Ordered field names that define the multi-field index schema."""
        ...

    def load_queries(self, data_root: Path, split: str) -> list[Any]:
        """Return query objects for the given split.

        Each object must expose .query_id, .query, .answer_ids attributes.
        """
        ...

    def build_corpus(self, data_root: Path, *, max_docs: int = -1) -> Any:
        """Return corpus object exposing .doc_ids and .documents."""
        ...

    def field_text(self, document: dict[str, Any], field_name: str) -> str:
        """Serialize one logical field of a document to indexable text."""
        ...


_REGISTRY: dict[str, DatasetSchema] = {}


def register_dataset(schema: DatasetSchema) -> None:
    """Register a dataset schema under its .name key."""
    _REGISTRY[schema.name] = schema


def get_dataset(name: str) -> DatasetSchema:
    """Retrieve a registered dataset schema by name.

    Raises KeyError when the name is unknown — ensure the dataset's
    __init__.py has been imported so auto-registration runs.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset: {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    """Return names of all registered datasets."""
    return sorted(_REGISTRY)


__all__ = ["DatasetSchema", "get_dataset", "list_datasets", "register_dataset"]
