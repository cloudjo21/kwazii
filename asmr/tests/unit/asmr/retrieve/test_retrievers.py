import pytest
from unittest.mock import Mock
from PIL import Image

from asmr.index.models import FieldBasedRanking, FieldBasedRankingItem
from asmr.retrieve.query import Query, QueryContent
from asmr.retrieve.retrievers import (
    ALLOW_RETRIEVE_MODE,
    DenseImageFieldRetriever,
    DenseTextFieldRetriever,
    MultiModalFieldRetriever,
    SparseTextFieldRetriever,
)


def make_ranking(field_name: str = "content", doc_id: str = "doc1", score: float = 1.0) -> FieldBasedRanking:
    return FieldBasedRanking(
        field_name=field_name,
        query="test",
        items=[FieldBasedRankingItem(doc_id=doc_id, score=score)],
        total_retrieved=1,
    )


def make_image() -> Image.Image:
    return Image.new("RGB", (10, 10))


class TestSparseTextFieldRetriever:
    def test_retrieve_with_string_query(self):
        mock_index = Mock()
        mock_index.search.return_value = make_ranking()
        retriever = SparseTextFieldRetriever(mock_index)

        result = retriever.retrieve("hello world", k=5)

        mock_index.search.assert_called_once_with("hello world", 5)
        assert result is mock_index.search.return_value

    def test_retrieve_with_query_object(self):
        mock_index = Mock()
        mock_index.search.return_value = make_ranking()
        retriever = SparseTextFieldRetriever(mock_index)
        query = Query.from_text("hello world")

        retriever.retrieve(query, k=3)

        mock_index.search.assert_called_once_with("hello world", 3)

    def test_retrieve_raises_on_image_only_query(self):
        mock_index = Mock()
        retriever = SparseTextFieldRetriever(mock_index)
        query = Query.from_image(make_image())

        with pytest.raises(ValueError):
            retriever.retrieve(query, k=5)

    def test_default_retrieve_mode_is_text_only(self):
        retriever = SparseTextFieldRetriever(Mock())

        assert retriever.retrieve_mode == ALLOW_RETRIEVE_MODE.TEXT_ONLY


class TestDenseTextFieldRetriever:
    def test_retrieve_with_string_query(self):
        mock_index = Mock()
        mock_index.search.return_value = make_ranking()
        retriever = DenseTextFieldRetriever(mock_index)

        retriever.retrieve("travel adventure", k=10)

        mock_index.search.assert_called_once_with("travel adventure", 10)

    def test_retrieve_with_query_object(self):
        mock_index = Mock()
        mock_index.search.return_value = make_ranking()
        retriever = DenseTextFieldRetriever(mock_index)
        query = Query.from_text("food experience")

        retriever.retrieve(query, k=5)

        mock_index.search.assert_called_once_with("food experience", 5)

    def test_retrieve_raises_on_image_only_query(self):
        mock_index = Mock()
        retriever = DenseTextFieldRetriever(mock_index)
        query = Query.from_image(make_image())

        with pytest.raises(ValueError):
            retriever.retrieve(query, k=5)

    def test_default_retrieve_mode_is_text_only(self):
        retriever = DenseTextFieldRetriever(Mock())

        assert retriever.retrieve_mode == ALLOW_RETRIEVE_MODE.TEXT_ONLY


class TestDenseImageFieldRetriever:
    def test_retrieve_text_to_image_with_string_query(self):
        mock_index = Mock()
        mock_index.search.return_value = make_ranking()
        retriever = DenseImageFieldRetriever(mock_index)

        retriever.retrieve("beautiful sunset", k=5)

        mock_index.search.assert_called_once_with("beautiful sunset", 5)

    def test_retrieve_image_to_image_with_image_query(self):
        mock_index = Mock()
        mock_index.search.return_value = make_ranking()
        retriever = DenseImageFieldRetriever(
            mock_index, retrieve_mode=ALLOW_RETRIEVE_MODE.IMAGE_ONLY
        )
        image = make_image()
        query = Query.from_image(image)

        retriever.retrieve(query, k=5)

        mock_index.search.assert_called_once_with(image, 5)

    def test_default_retrieve_mode_is_image_only(self):
        retriever = DenseImageFieldRetriever(Mock())

        assert retriever.retrieve_mode == ALLOW_RETRIEVE_MODE.IMAGE_ONLY


class TestMultiModalFieldRetriever:
    def test_retrieve_with_string_query_uses_text_index(self):
        mock_text_index = Mock()
        mock_text_index.search.return_value = make_ranking()
        mock_image_index = Mock()
        retriever = MultiModalFieldRetriever(mock_text_index, mock_image_index)

        retriever.retrieve("travel adventure", k=5)

        mock_text_index.search.assert_called_once_with("travel adventure", 5)
        mock_image_index.search.assert_not_called()

    def test_retrieve_raises_on_text_only_query(self):
        retriever = MultiModalFieldRetriever(Mock(), Mock())
        text_only_query = Query.from_text("only text")

        with pytest.raises(ValueError):
            retriever.retrieve(text_only_query, k=5)

    def test_retrieve_raises_on_image_only_query(self):
        retriever = MultiModalFieldRetriever(Mock(), Mock())
        image_only_query = Query.from_image(make_image())

        with pytest.raises(ValueError):
            retriever.retrieve(image_only_query, k=5)

    def test_retrieve_with_multimodal_query_calls_both_indices(self):
        mock_text_index = Mock()
        mock_text_index.search.return_value = make_ranking("title", "doc1", 2.0)
        mock_image_index = Mock()
        mock_image_index.search.return_value = make_ranking("image", "doc2", 1.0)
        retriever = MultiModalFieldRetriever(mock_text_index, mock_image_index)
        image = make_image()
        query = Query.from_multimodal("sunset view", image)

        result = retriever.retrieve(query, k=5)

        mock_text_index.search.assert_called_once_with("sunset view", 5)
        mock_image_index.search.assert_called_once_with(image, 5)

    def test_multimodal_results_sorted_by_score_descending(self):
        mock_text_index = Mock()
        mock_text_index.search.return_value = make_ranking("title", "doc_text", 0.5)
        mock_image_index = Mock()
        mock_image_index.search.return_value = make_ranking("image", "doc_image", 2.0)
        retriever = MultiModalFieldRetriever(mock_text_index, mock_image_index)
        query = Query.from_multimodal("query", make_image())

        result = retriever.retrieve(query, k=5)

        # Higher score should come first
        assert result[0][1] >= result[1][1]

    def test_default_retrieve_mode_is_multimodal(self):
        retriever = MultiModalFieldRetriever(Mock(), Mock())

        assert retriever.retrieve_mode == ALLOW_RETRIEVE_MODE.MULTIMODAL
