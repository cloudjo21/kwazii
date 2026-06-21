"""Unit tests for StarkAmazonSchema — no file I/O."""

from typing import Any

import pytest

from asmr.datasets import DatasetSchema
from asmr.datasets.stark_amazon.loader import (
    AMAZON_FIELD_NAMES,
    AmazonCorpus,
    AmazonQuery,
    field_text,
)
from asmr.datasets.stark_amazon.schema import STARK_AMAZON_SCHEMA, StarkAmazonSchema


class TestStarkAmazonSchemaMetadata:
    def test_name(self) -> None:
        assert STARK_AMAZON_SCHEMA.name == "stark_amazon"

    def test_field_names_count(self) -> None:
        assert len(STARK_AMAZON_SCHEMA.field_names) == 8

    def test_field_names_content(self) -> None:
        assert STARK_AMAZON_SCHEMA.field_names == AMAZON_FIELD_NAMES

    def test_protocol_compliance(self) -> None:
        assert isinstance(STARK_AMAZON_SCHEMA, DatasetSchema)


class TestAmazonFieldText:
    def _doc(self) -> dict[str, Any]:
        return {
            "title": "Chess Strategy Guide",
            "brand": "House of Staunton",
            "description": "A comprehensive chess guide",
            "feature": ["Hardcover", "300 pages"],
            "review": "Excellent book",
            "qa": "Q: Is this good? A: Yes",
            "also_buy": ["B001", "B002"],
            "also_view": ["B003"],
        }

    def test_string_field(self) -> None:
        assert field_text(self._doc(), "title") == "Chess Strategy Guide"

    def test_string_field_brand(self) -> None:
        assert field_text(self._doc(), "brand") == "House of Staunton"

    def test_list_field(self) -> None:
        result = field_text(self._doc(), "feature")
        assert "Hardcover" in result
        assert "300 pages" in result

    def test_list_field_also_buy(self) -> None:
        result = field_text(self._doc(), "also_buy")
        assert "B001" in result
        assert "B002" in result

    def test_missing_field_returns_empty(self) -> None:
        assert field_text({}, "title") == ""

    def test_schema_field_text_delegates(self) -> None:
        doc = self._doc()
        assert STARK_AMAZON_SCHEMA.field_text(doc, "title") == field_text(doc, "title")


class TestAmazonQueryModel:
    def test_valid_query(self) -> None:
        q = AmazonQuery(query_id=1, query="Find a chess book", answer_ids=["B001"])
        assert q.query_id == 1
        assert q.answer_ids == ["B001"]

    def test_frozen(self) -> None:
        q = AmazonQuery(query_id=1, query="test", answer_ids=[])
        with pytest.raises(Exception):
            q.query_id = 2  # type: ignore[misc]


class TestAmazonCorpusModel:
    def test_valid_corpus(self) -> None:
        corpus = AmazonCorpus(doc_ids=["B001"], documents=[{"title": "Chess Guide"}])
        assert len(corpus.doc_ids) == 1

    def test_frozen(self) -> None:
        corpus = AmazonCorpus(doc_ids=[], documents=[])
        with pytest.raises(Exception):
            corpus.doc_ids = ["X"]  # type: ignore[misc]


class TestStarkAmazonSchemaInstance:
    def test_singleton_is_schema_instance(self) -> None:
        assert isinstance(STARK_AMAZON_SCHEMA, StarkAmazonSchema)
