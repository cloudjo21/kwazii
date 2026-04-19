import abc
import enum
from typing import TYPE_CHECKING, Union

from asmr.index import fields
from asmr.index.models import FieldBasedRanking, FieldBasedRankingItem
from asmr.retrieve.query import Query
from asmr.retrieve.result import Err, Ok, Result

if TYPE_CHECKING:
    from PIL import Image

TOP_K_RETRIEVE = 100


class ALLOW_RETRIEVE_MODE(enum.Enum):
    """Deprecated: use BaseFieldRetriever.can_handle() instead."""

    TEXT_ONLY = 1
    IMAGE_ONLY = 2
    MULTIMODAL = 3


# ---------------------------------------------------------------------------
# Module-level pure helpers — single coercion point for query content
# ---------------------------------------------------------------------------


def _extract_text_content(query: Union[str, Query]) -> str:
    if isinstance(query, str):
        return query
    if not query.has_text():
        raise ValueError("Query has no text content")
    return query.get_text()


def _extract_image_content(query: Query) -> "Image.Image":
    if not query.has_image():
        raise ValueError("Query has no image content")
    return query.get_image()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseFieldRetriever(abc.ABC):
    """Single-field retriever strategy."""

    @abc.abstractmethod
    def can_handle(self, query: Query) -> bool:
        """Return True if this retriever is compatible with the query's content."""
        ...

    @abc.abstractmethod
    def retrieve(self, query: Query, k: int) -> Result[FieldBasedRanking, str]:
        """Retrieve top-k results.

        Returns Ok(ranking) on success or Err(reason) when the query type is
        incompatible. Raises only for programming errors.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete retrievers
# ---------------------------------------------------------------------------


class SparseTextFieldRetriever(BaseFieldRetriever):
    """BM25-style sparse text retriever."""

    def __init__(
        self,
        field_index: fields.SparseTextFieldIndex,
        retrieve_mode: ALLOW_RETRIEVE_MODE = ALLOW_RETRIEVE_MODE.TEXT_ONLY,
    ):
        self.field_index = field_index
        self.retrieve_mode = retrieve_mode

    def can_handle(self, query: Query) -> bool:
        return query.has_text()

    def retrieve(self, query: Query, k: int) -> Result[FieldBasedRanking, str]:
        if not self.can_handle(query):
            return Err("SparseTextFieldRetriever requires text content in query")
        return Ok(self.field_index.search(_extract_text_content(query), k))


class DenseTextFieldRetriever(BaseFieldRetriever):
    """Dense text embedding retriever."""

    def __init__(
        self,
        field_index: fields.DenseTextFieldIndex,
        retrieve_mode: ALLOW_RETRIEVE_MODE = ALLOW_RETRIEVE_MODE.TEXT_ONLY,
    ):
        self.field_index = field_index
        self.retrieve_mode = retrieve_mode

    def can_handle(self, query: Query) -> bool:
        return query.has_text()

    def retrieve(self, query: Query, k: int) -> Result[FieldBasedRanking, str]:
        if not self.can_handle(query):
            return Err("DenseTextFieldRetriever requires text content in query")
        return Ok(self.field_index.search(_extract_text_content(query), k))


class DenseImageFieldRetriever(BaseFieldRetriever):
    """Dense image retriever; supports both image-to-image and text-to-image search."""

    def __init__(
        self,
        field_index: fields.DenseImageFieldIndex,
        retrieve_mode: ALLOW_RETRIEVE_MODE = ALLOW_RETRIEVE_MODE.IMAGE_ONLY,
    ):
        self.field_index = field_index
        self.retrieve_mode = retrieve_mode

    def can_handle(self, query: Query) -> bool:
        return query.has_image() or query.has_text()

    def retrieve(self, query: Query, k: int) -> Result[FieldBasedRanking, str]:
        if not self.can_handle(query):
            return Err("Query must have either text or image content")
        # Image takes priority; fall back to text for cross-modal search.
        query_content = (
            _extract_image_content(query)
            if query.has_image()
            else _extract_text_content(query)
        )
        return Ok(self.field_index.search(query_content, k))


class MultiModalFieldRetriever(BaseFieldRetriever):
    """Retriever for queries that carry both text and image content."""

    def __init__(
        self,
        text_field_index: fields.DenseTextFieldIndex,
        image_field_index: fields.DenseImageFieldIndex,
        retrieve_mode: ALLOW_RETRIEVE_MODE = ALLOW_RETRIEVE_MODE.MULTIMODAL,
    ):
        self.text_field_index = text_field_index
        self.image_field_index = image_field_index
        self.retrieve_mode = retrieve_mode

    def can_handle(self, query: Query) -> bool:
        return query.has_text() and query.has_image()

    def retrieve(self, query: Query, k: int) -> Result[FieldBasedRanking, str]:
        if not self.can_handle(query):
            return Err(
                "MultiModalFieldRetriever requires both text and image content in query"
            )

        text_ranking: FieldBasedRanking = self.text_field_index.search(
            _extract_text_content(query), k
        )
        image_ranking: FieldBasedRanking = self.image_field_index.search(
            _extract_image_content(query), k
        )

        # Merge and re-rank by score; keep top-k as a proper FieldBasedRanking.
        merged = sorted(
            list(text_ranking) + list(image_ranking),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        items = [
            FieldBasedRankingItem(doc_id=doc_id, score=score)
            for doc_id, score in merged
        ]
        return Ok(
            FieldBasedRanking(
                field_name="multimodal",
                query=str(query),
                items=items,
                total_retrieved=len(items),
            )
        )


# ---------------------------------------------------------------------------
# Router — single-field registry (low-level dispatcher)
# ---------------------------------------------------------------------------


class QueryRouter:
    """Single-field registry and dispatcher.

    Responsibilities:
    - Map field names to their BaseFieldRetriever implementations.
    - Dispatch a single-field retrieve call, returning Ok/Err based on
      can_handle() compatibility.

    Does NOT decide which fields to query; that policy lives in
    DocumentRetriever (helpers.py) and pipeline.py.
    """

    def __init__(self, field_retrievers: dict[str, BaseFieldRetriever]) -> None:
        self.field_retrievers = field_retrievers

    @property
    def fields(self) -> list[str]:
        return list(self.field_retrievers.keys())

    def retrieve(
        self, field: str, query: Query, k: int
    ) -> Result[FieldBasedRanking, str]:
        if field not in self.field_retrievers:
            raise KeyError(f"Field '{field}' not found in QueryRouter.")
        retriever = self.field_retrievers[field]
        if not retriever.can_handle(query):
            return Err(f"Retriever for '{field}' cannot handle this query type.")
        return retriever.retrieve(query, k)
