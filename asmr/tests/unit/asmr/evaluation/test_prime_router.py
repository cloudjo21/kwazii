"""Unit tests for STaRK-Prime router and BM25 search_topk integration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from asmr.evaluation.stark_prime_router import attach_field_mask
from asmr.index.bm25 import BM25Index, DocumentIndexToIdMapping, DocIdPostingPolicy
from asmr.vocab import Vocabulary


def _make_index(texts: list[str], doc_ids: list[str]) -> BM25Index:
    tokens: set[str] = set()
    for text in texts:
        tokens.update(text.lower().split())
    vocab = Vocabulary.from_token_set(tokens)
    n_docs = len(texts)
    n_vocab = len(vocab)
    mat = np.zeros((n_docs, n_vocab), dtype=np.float32)
    for doc_pos, text in enumerate(texts):
        for token in text.lower().split():
            if token in vocab:
                mat[doc_pos, vocab.id(token)] = 1.0 + doc_pos
    mapping = DocumentIndexToIdMapping.build(
        enumerate(doc_ids),
        DocIdPostingPolicy.UNIQUE,
    )
    return BM25Index("field", csc_matrix(mat), vocab, mapping)


class TestAttachFieldMask:
    def test_mask_shape_matches_scores(self) -> None:
        class _Store:
            field_names = ["f0", "f1"]
            id_to_col = {"d0": 0, "d1": 1}
            field_mask = np.array([[True, False], [False, True]])

        store = _Store()
        doc_ids = ["d0", "d1"]
        scores = np.zeros((2, 2, 2), dtype=np.float32)
        mask = attach_field_mask(store, doc_ids, scores)  # type: ignore[arg-type]
        assert mask.shape == (2, 2)
        assert mask[0, 0]
        assert not mask[0, 1]
        assert not mask[1, 0]
        assert mask[1, 1]


class TestSearchTopkIntegration:
    def test_search_topk_returns_ranked_doc_ids(self) -> None:
        texts = ["alpha beta", "beta gamma", "alpha gamma"]
        doc_ids = ["a", "b", "c"]
        index = _make_index(texts, doc_ids)
        hits = index.search_topk(["alpha"], k=2)
        assert len(hits) <= 2
        assert all(isinstance(doc_id, str) for doc_id, _ in hits)
        scores = [score for _, score in hits]
        assert scores == sorted(scores, reverse=True)

    def test_search_topk_requires_mapping(self) -> None:
        texts = ["hello"]
        vocab = Vocabulary.from_token_set({"hello"})
        mat = np.array([[1.0]], dtype=np.float32)
        index = BM25Index("f", csc_matrix(mat), vocab, None)
        with pytest.raises(ValueError, match="doc_id_mapping"):
            index.search_topk(["hello"], k=1)
