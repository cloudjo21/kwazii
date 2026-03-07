import pytest
import numpy as np

from asmr.vocab import Vocabulary
from asmr.tokenize.helpers import SplitTokenizer, TokenizerWrapper
from asmr.columnar import (
    ColumnarTextStream,
    FieldBasedColumnarTexts,
    ColumnarStatistics,
    ColumnarStatisticsBuilder,
)


def make_vocabulary(texts: list[str]) -> Vocabulary:
    all_tokens: set[str] = set()
    for text in texts:
        all_tokens.update(text.lower().split())
    return Vocabulary.from_token_set(all_tokens)


def make_field_texts(
    docs: list[dict], field_name: str
) -> FieldBasedColumnarTexts:
    return FieldBasedColumnarTexts(docs, field_name, len(docs))


class TestColumnarTextStream:
    def test_tokenization_yields_token_id_arrays(self):
        vocabulary = make_vocabulary(["hello world"])
        tokenizer = TokenizerWrapper(SplitTokenizer())
        stream = ColumnarTextStream(["hello world"], vocabulary.trie, tokenizer)

        results = list(stream)

        assert len(results) == 1
        assert results[0].dtype == np.int32
        assert len(results[0]) == 2

    def test_multiple_texts_yield_separate_arrays(self):
        vocabulary = make_vocabulary(["hello world", "test"])
        tokenizer = TokenizerWrapper(SplitTokenizer())
        stream = ColumnarTextStream(
            ["hello world", "test"], vocabulary.trie, tokenizer
        )

        results = list(stream)

        assert len(results) == 2
        assert len(results[0]) == 2  # "hello", "world"
        assert len(results[1]) == 1  # "test"

    def test_unknown_tokens_are_filtered(self):
        vocabulary = make_vocabulary(["hello"])
        tokenizer = TokenizerWrapper(SplitTokenizer())
        stream = ColumnarTextStream(["hello unknown_word"], vocabulary.trie, tokenizer)

        results = list(stream)

        # "unknown_word" not in vocab → only "hello" survives
        assert len(results[0]) == 1

    def test_empty_text_yields_empty_array(self):
        vocabulary = make_vocabulary(["hello"])
        tokenizer = TokenizerWrapper(SplitTokenizer())
        stream = ColumnarTextStream([""], vocabulary.trie, tokenizer)

        results = list(stream)

        assert len(results[0]) == 0

    def test_token_ids_are_valid_indices(self):
        vocabulary = make_vocabulary(["apple banana"])
        tokenizer = TokenizerWrapper(SplitTokenizer())
        stream = ColumnarTextStream(["apple banana"], vocabulary.trie, tokenizer)

        results = list(stream)
        token_ids = results[0]

        assert all(0 <= tid < len(vocabulary) for tid in token_ids)


class TestFieldBasedColumnarTexts:
    def test_field_extraction(self):
        documents = [
            {"title": "first doc", "content": "content one"},
            {"title": "second doc", "content": "content two"},
        ]
        columnar = FieldBasedColumnarTexts(documents, "title", 2)

        texts = list(columnar)

        assert texts == ["first doc", "second doc"]

    def test_nested_field_extraction(self):
        documents = [
            {"review": "great place"},
            {"review": "loved the food"},
        ]
        columnar = FieldBasedColumnarTexts(documents, "review", 2)

        texts = list(columnar)

        assert texts == ["great place", "loved the food"]

    def test_n_docs_attribute(self):
        documents = [{"field": "text"}]
        columnar = FieldBasedColumnarTexts(documents, "field", 5)

        assert columnar.n_docs == 5


class TestColumnarStatisticsBuilder:
    def test_columnar_statistics_calculation(self):
        docs = [
            {"content": "hello world"},
            {"content": "hello test"},
            {"content": "world test"},
        ]
        vocabulary = make_vocabulary([d["content"] for d in docs])
        field_texts = make_field_texts(docs, "content")

        stats = ColumnarStatisticsBuilder.build(field_texts, vocabulary)

        assert stats.n_docs == 3
        assert len(stats.df) == len(vocabulary)
        assert len(stats.idf) == len(vocabulary)
        assert len(stats.doc_len) == 3
        assert stats.avg_doc_len == pytest.approx(2.0)
        assert stats.nnz_total > 0

    def test_statistics_builder_with_empty_vocab(self):
        docs = [{"content": "hello"}]
        empty_vocab = Vocabulary.from_token_set(set())
        field_texts = make_field_texts(docs, "content")

        with pytest.raises(ValueError, match="Vocabulary is not provided or invalid"):
            ColumnarStatisticsBuilder.build(field_texts, empty_vocab)

    def test_statistics_builder_edge_cases_single_doc(self):
        docs = [{"content": "only one document"}]
        vocabulary = make_vocabulary([d["content"] for d in docs])
        field_texts = make_field_texts(docs, "content")

        stats = ColumnarStatisticsBuilder.build(field_texts, vocabulary)

        assert stats.n_docs == 1
        assert stats.avg_doc_len == pytest.approx(3.0)

    def test_idf_is_non_negative(self):
        docs = [
            {"content": "hello world"},
            {"content": "hello test"},
        ]
        vocabulary = make_vocabulary([d["content"] for d in docs])
        field_texts = make_field_texts(docs, "content")

        stats = ColumnarStatisticsBuilder.build(field_texts, vocabulary)

        assert np.all(stats.idf >= 0)

    def test_df_counts_document_frequency(self):
        docs = [
            {"content": "hello world"},
            {"content": "hello test"},
            {"content": "world only"},
        ]
        vocabulary = make_vocabulary([d["content"] for d in docs])
        field_texts = make_field_texts(docs, "content")

        stats = ColumnarStatisticsBuilder.build(field_texts, vocabulary)

        hello_id = vocabulary.id("hello")
        world_id = vocabulary.id("world")

        # "hello" appears in 2 docs, "world" appears in 2 docs
        assert stats.df[hello_id] == 2
        assert stats.df[world_id] == 2

    def test_doc_len_reflects_token_count(self):
        docs = [
            {"content": "one"},
            {"content": "one two"},
            {"content": "one two three"},
        ]
        vocabulary = make_vocabulary([d["content"] for d in docs])
        field_texts = make_field_texts(docs, "content")

        stats = ColumnarStatisticsBuilder.build(field_texts, vocabulary)

        assert list(stats.doc_len) == [1, 2, 3]
        assert stats.avg_doc_len == pytest.approx(2.0)

    def test_statistics_field_name_is_preserved(self):
        docs = [{"body": "some text"}]
        vocabulary = make_vocabulary(["some text"])
        field_texts = make_field_texts(docs, "body")

        stats = ColumnarStatisticsBuilder.build(field_texts, vocabulary)

        assert stats.field_name == "body"
