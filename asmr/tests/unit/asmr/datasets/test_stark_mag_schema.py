"""Unit tests for StarkMagSchema — no file I/O."""

from typing import Any

import pytest

from asmr.datasets import DatasetSchema
from asmr.datasets.stark_mag.loader import (
    MAG_FIELD_NAMES,
    MagCorpus,
    MagQuery,
    field_text,
)
from asmr.datasets.stark_mag.schema import STARK_MAG_SCHEMA, StarkMagSchema


class TestStarkMagSchemaMetadata:
    def test_name(self) -> None:
        assert STARK_MAG_SCHEMA.name == "stark_mag"

    def test_field_names_count(self) -> None:
        assert len(STARK_MAG_SCHEMA.field_names) == 5

    def test_field_names_content(self) -> None:
        assert STARK_MAG_SCHEMA.field_names == MAG_FIELD_NAMES

    def test_protocol_compliance(self) -> None:
        assert isinstance(STARK_MAG_SCHEMA, DatasetSchema)


class TestMagFieldText:
    def _doc(self) -> dict[str, Any]:
        return {
            "title": "Energy Levels of Fe II",
            "abstract": "We study configuration interaction in iron.",
            "author___affiliated_with___institution": ["Indian Maritime University"],
            "paper___cites___paper": ["paper_001", "paper_002"],
            "paper___has_topic___field_of_study": ["Atomic Physics", "Spectroscopy"],
        }

    def test_string_field_title(self) -> None:
        assert field_text(self._doc(), "title") == "Energy Levels of Fe II"

    def test_string_field_abstract(self) -> None:
        result = field_text(self._doc(), "abstract")
        assert "configuration interaction" in result

    def test_list_field_institution(self) -> None:
        result = field_text(self._doc(), "author___affiliated_with___institution")
        assert "Indian Maritime University" in result

    def test_list_field_cites(self) -> None:
        result = field_text(self._doc(), "paper___cites___paper")
        assert "paper_001" in result
        assert "paper_002" in result

    def test_list_field_topic(self) -> None:
        result = field_text(self._doc(), "paper___has_topic___field_of_study")
        assert "Atomic Physics" in result

    def test_missing_field_returns_empty(self) -> None:
        assert field_text({}, "abstract") == ""

    def test_schema_field_text_delegates(self) -> None:
        doc = self._doc()
        assert STARK_MAG_SCHEMA.field_text(doc, "title") == field_text(doc, "title")


class TestMagQueryModel:
    def test_valid_query(self) -> None:
        q = MagQuery(query_id=42, query="Find Fe II energy papers", answer_ids=["P001"])
        assert q.query_id == 42
        assert q.answer_ids == ["P001"]

    def test_frozen(self) -> None:
        q = MagQuery(query_id=1, query="test", answer_ids=[])
        with pytest.raises(Exception):
            q.query_id = 2  # type: ignore[misc]


class TestMagCorpusModel:
    def test_valid_corpus(self) -> None:
        corpus = MagCorpus(doc_ids=["P001"], documents=[{"title": "Fe II Paper"}])
        assert len(corpus.doc_ids) == 1

    def test_frozen(self) -> None:
        corpus = MagCorpus(doc_ids=[], documents=[])
        with pytest.raises(Exception):
            corpus.doc_ids = ["X"]  # type: ignore[misc]


class TestStarkMagSchemaInstance:
    def test_singleton_is_schema_instance(self) -> None:
        assert isinstance(STARK_MAG_SCHEMA, StarkMagSchema)
