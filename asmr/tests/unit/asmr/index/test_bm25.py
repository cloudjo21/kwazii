import pytest
import numpy as np
from scipy.sparse import csc_matrix

import asmr.index.bm25 as bm25_module
from asmr.vocab import Vocabulary
from asmr.columnar import (
    FieldBasedColumnarTexts,
    ColumnarStatisticsBuilder,
)
from asmr.index.bm25 import (
    BM25Index,
    BM25Indexer,
    DocumentIndexToIdMapping,
    DocIdPostingPolicy,
    DocumentScore,
    TermScore,
)


def make_vocabulary(texts: list[str]) -> Vocabulary:
    all_tokens: set[str] = set()
    for text in texts:
        all_tokens.update(text.lower().split())
    return Vocabulary.from_token_set(all_tokens)


def make_bm25_index(
    field_name: str,
    texts: list[str],
    vocabulary: Vocabulary,
) -> BM25Index:
    """Build a BM25Index directly from a dense matrix for unit testing."""
    n_docs = len(texts)
    n_vocab = len(vocabulary)
    mat = np.zeros((n_docs, n_vocab), dtype=np.float32)

    for doc_id, text in enumerate(texts):
        for token in text.lower().split():
            if token in vocabulary:
                mat[doc_id, vocabulary.id(token)] += 1.0

    return BM25Index(field_name, csc_matrix(mat), vocabulary)


class TestBM25Index:
    def test_bm25_index_creation(self):
        texts = ["hello world", "test document"]
        vocabulary = make_vocabulary(texts)
        index = make_bm25_index("content", texts, vocabulary)

        assert index.field_name == "content"
        assert index.vocab is vocabulary
        assert index.index.shape == (2, len(vocabulary))

    def test_bm25_index_score_calculation(self):
        texts = ["hello world", "test document"]
        vocabulary = make_vocabulary(texts)
        mat = np.zeros((2, len(vocabulary)), dtype=np.float32)
        mat[0, vocabulary.id("hello")] = 1.5
        mat[0, vocabulary.id("world")] = 2.0
        index = BM25Index("content", csc_matrix(mat), vocabulary)

        doc_score = index.get_score(0, ["hello", "world"])

        assert isinstance(doc_score, DocumentScore)
        assert doc_score.doc_id == 0
        assert len(doc_score.term_scores) == 2
        score_map = {ts.term: ts.score for ts in doc_score.term_scores}
        assert score_map["hello"] == pytest.approx(1.5)
        assert score_map["world"] == pytest.approx(2.0)

    def test_bm25_index_unknown_term_ignored(self):
        texts = ["hello world"]
        vocabulary = make_vocabulary(texts)
        index = make_bm25_index("content", texts, vocabulary)

        doc_score = index.get_score(0, ["hello", "nonexistent"])

        terms = [ts.term for ts in doc_score.term_scores]
        assert "hello" in terms
        assert "nonexistent" not in terms

    def test_bm25_index_invalid_doc_id_raises(self):
        texts = ["hello world"]
        vocabulary = make_vocabulary(texts)
        index = make_bm25_index("content", texts, vocabulary)

        with pytest.raises(ValueError, match="not found in index"):
            index.get_score(99, ["hello"])

    def test_bm25_index_negative_doc_id_raises(self):
        texts = ["hello world"]
        vocabulary = make_vocabulary(texts)
        index = make_bm25_index("content", texts, vocabulary)

        with pytest.raises(ValueError, match="not found in index"):
            index.get_score(-1, ["hello"])

    def test_bm25_index_zero_score_for_absent_term(self):
        texts = ["hello world", "test only"]
        vocabulary = make_vocabulary(texts)
        index = make_bm25_index("content", texts, vocabulary)

        # doc 1 has no "hello" → score should be 0.0
        doc_score = index.get_score(1, ["hello"])

        assert len(doc_score.term_scores) == 1
        assert doc_score.term_scores[0].score == pytest.approx(0.0)


class TestBM25Indexer:
    def test_bm25_indexer_build_process(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bm25_module, "_INDEX_DIR", str(tmp_path))
        texts = ["hello world", "test hello", "world test"]
        vocabulary = make_vocabulary(texts)
        docs = [{"content": t} for t in texts]
        field_texts = FieldBasedColumnarTexts(docs, "content", len(docs))
        stats = ColumnarStatisticsBuilder.build(field_texts, vocabulary)

        result = BM25Indexer.build(
            field_name="content",
            columnar_posting=texts,
            vocab=vocabulary,
            field_statistics=stats,
        )

        assert isinstance(result, BM25Index)
        assert result.field_name == "content"
        assert result.index.shape == (len(texts), len(vocabulary))

    def test_bm25_indexer_build_with_doc_ids(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bm25_module, "_INDEX_DIR", str(tmp_path))
        texts = ["hello world", "test hello"]
        doc_ids = ["doc_a", "doc_b"]
        vocabulary = make_vocabulary(texts)
        docs = [{"content": t} for t in texts]
        field_texts = FieldBasedColumnarTexts(docs, "content", len(docs))
        stats = ColumnarStatisticsBuilder.build(field_texts, vocabulary)

        result = BM25Indexer.build(
            field_name="content",
            columnar_posting=texts,
            vocab=vocabulary,
            field_statistics=stats,
            doc_ids=doc_ids,
        )

        assert result.doc_id_mapping is not None
        assert result.doc_id_mapping.get_doc_id(0) == "doc_a"
        assert result.doc_id_mapping.get_doc_id(1) == "doc_b"

    def test_bm25_index_persistence(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bm25_module, "_INDEX_DIR", str(tmp_path))
        texts = ["hello world", "world hello"]
        doc_ids = ["doc_a", "doc_b"]
        vocabulary = make_vocabulary(texts)
        docs = [{"content": t} for t in texts]
        field_texts = FieldBasedColumnarTexts(docs, "content", len(docs))
        stats = ColumnarStatisticsBuilder.build(field_texts, vocabulary)

        BM25Indexer.build(
            "content",
            texts,
            vocabulary,
            stats,
            doc_ids=doc_ids,
            index_dir=tmp_path,
        )
        loaded = BM25Indexer.load("content", vocabulary, index_dir=tmp_path)

        assert isinstance(loaded, BM25Index)
        assert loaded.index.shape == (len(texts), len(vocabulary))
        assert loaded.doc_id_mapping is not None
        assert loaded.doc_id_mapping.get_doc_id(0) == "doc_a"

    def test_bm25_index_search_topk_matches_brute_force(self):
        texts = [
            "alpha beta gamma",
            "beta delta",
            "gamma epsilon",
            "alpha delta epsilon",
        ]
        doc_ids = [f"doc_{i}" for i in range(len(texts))]
        vocabulary = make_vocabulary(texts)
        mat = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
        for doc_pos, text in enumerate(texts):
            for token in text.lower().split():
                if token in vocabulary:
                    mat[doc_pos, vocabulary.id(token)] += 1.0 + doc_pos * 0.1

        mapping = DocumentIndexToIdMapping.build(
            enumerate(doc_ids),
            DocIdPostingPolicy.UNIQUE,
        )
        index = BM25Index("content", csc_matrix(mat), vocabulary, mapping)

        queries = [
            ["alpha"],
            ["beta", "gamma"],
            ["epsilon"],
            ["missing"],
        ]
        for terms in queries:
            k = 3
            topk_ids = {doc_id for doc_id, _ in index.search_topk(terms, k)}
            brute: list[tuple[str, float]] = []
            for doc_pos in range(index.index.shape[0]):
                doc_score = index.get_score(doc_pos, terms)
                total = sum(ts.score for ts in doc_score.term_scores)
                if total > 0:
                    brute.append((mapping.get_doc_id(doc_pos), total))
            brute.sort(key=lambda item: item[1], reverse=True)
            brute_ids = {doc_id for doc_id, _ in brute[:k]}
            assert topk_ids == brute_ids

    def test_bm25_indexer_load_missing_files_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bm25_module, "_INDEX_DIR", str(tmp_path))
        vocabulary = make_vocabulary(["hello"])

        with pytest.raises(FileNotFoundError):
            BM25Indexer.load("content", vocabulary)


class TestDocumentIndexToIdMapping:
    def test_build_unique_policy(self):
        pairs = [(0, "doc_a"), (1, "doc_b"), (2, "doc_c")]
        mapping = DocumentIndexToIdMapping.build(pairs, DocIdPostingPolicy.UNIQUE)

        assert mapping.get_doc_id(0) == "doc_a"
        assert mapping.get_doc_id(1) == "doc_b"
        assert mapping.get_doc_id(2) == "doc_c"

    def test_build_unique_policy_raises_on_duplicate(self):
        pairs = [(0, "doc_a"), (0, "doc_b")]

        with pytest.raises(ValueError, match="duplicate key"):
            DocumentIndexToIdMapping.build(pairs, DocIdPostingPolicy.UNIQUE)

    def test_build_first_win_policy(self):
        pairs = [(0, "first"), (0, "second")]
        mapping = DocumentIndexToIdMapping.build(pairs, DocIdPostingPolicy.FIRST_WIN)

        assert mapping.get_doc_id(0) == "first"

    def test_get_doc_id_missing_key_raises(self):
        pairs = [(0, "doc_a")]
        mapping = DocumentIndexToIdMapping.build(pairs, DocIdPostingPolicy.UNIQUE)

        with pytest.raises(KeyError):
            mapping.get_doc_id(99)
