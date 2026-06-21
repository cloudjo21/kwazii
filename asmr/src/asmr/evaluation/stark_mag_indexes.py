"""In-memory field indexes for STaRK-MAG MFARAll (5 fields × 2 scorers)."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from asmr.datasets.stark_mag.loader import (
    MAG_FIELD_NAMES,
    MagCorpus,
    MagQuery,
    field_text,
    single_field_text,
)
from asmr.datasets.stark_prime.torch_dataset import StarkRankingExample
from asmr.encode.protocol import TextEncoderProtocol
from asmr.index.config import FieldConfig, IndexUsage, RepresentationType, TokenizerType
from asmr.index.fields import DenseTextFieldIndex, SparseTextFieldIndex

logger = logging.getLogger(__name__)


class MagFieldIndexes:
    """Per-field BM25 and FAISS dense indexes for STaRK-MAG MFARAll."""

    def __init__(
        self,
        corpus: MagCorpus,
        encoder: TextEncoderProtocol,
        batch_size: int = 64,
    ) -> None:
        self.doc_ids = corpus.doc_ids
        self.id_to_col = {d: i for i, d in enumerate(self.doc_ids)}
        self.num_docs = len(self.doc_ids)
        self.field_names = list(MAG_FIELD_NAMES)
        self.field_mask = np.zeros((len(self.field_names), self.num_docs), dtype=bool)
        self._tmpdir = tempfile.TemporaryDirectory()
        self._sparse: list[Optional[SparseTextFieldIndex]] = []
        self._dense: list[DenseTextFieldIndex] = []

        for fi, fname in enumerate(self.field_names):
            texts = [field_text(doc, fname) for doc in corpus.documents]
            nonempty = [bool(t.strip()) for t in texts]
            self.field_mask[fi] = np.array(nonempty, dtype=bool)
            nonempty_texts = [t if t.strip() else fname for t in texts]

            sparse_cfg = FieldConfig(
                fname, TokenizerType.SPLIT, RepresentationType.SPARSE
            )
            sparse = SparseTextFieldIndex(sparse_cfg)
            if any(nonempty):
                sparse.add_documents(
                    corpus.doc_ids,
                    texts,
                    index_dir=Path(self._tmpdir.name) / f"sparse_{fname}",
                )
                self._sparse.append(sparse)
            else:
                self._sparse.append(None)

            dense_cfg = FieldConfig(
                fname,
                TokenizerType.SPLIT,
                RepresentationType.DENSE,
                usage=IndexUsage.BENCHMARK,
            )
            dense = DenseTextFieldIndex(dense_cfg, encoder)
            dense.add_documents(corpus.doc_ids, nonempty_texts)
            self._dense.append(dense)

        single_texts = [single_field_text(doc) for doc in corpus.documents]
        single_sparse_cfg = FieldConfig(
            "single", TokenizerType.SPLIT, RepresentationType.SPARSE
        )
        self._single_sparse = SparseTextFieldIndex(single_sparse_cfg)
        self._single_sparse.add_documents(
            corpus.doc_ids,
            single_texts,
            index_dir=Path(self._tmpdir.name) / "sparse_single",
        )
        single_dense_cfg = FieldConfig(
            "single",
            TokenizerType.SPLIT,
            RepresentationType.DENSE,
            usage=IndexUsage.BENCHMARK,
        )
        self._single_dense = DenseTextFieldIndex(single_dense_cfg, encoder)
        self._single_dense.add_documents(corpus.doc_ids, single_texts)

    def shortlist_hybrid(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Build MFARAll shortlist: union of per-field top-k (lex + dense).

        Returns:
            doc_ids: list of candidate doc IDs
            scores: float32 array [F, 2, D]
            mask: bool array [F, D]
        """
        f_num = len(self.field_names)
        union: dict[str, np.ndarray] = {}

        for fi in range(f_num):
            if self._sparse[fi] is not None:
                for item in self._sparse[fi].search(query_text, k).items:
                    union.setdefault(
                        item.doc_id, np.zeros((f_num, 2), dtype=np.float32)
                    )
                    union[item.doc_id][fi, 0] = max(
                        union[item.doc_id][fi, 0], item.score
                    )
            for item in self._dense[fi].search(query_text, k).items:
                union.setdefault(item.doc_id, np.zeros((f_num, 2), dtype=np.float32))
                union[item.doc_id][fi, 1] = max(union[item.doc_id][fi, 1], item.score)

        if not union:
            return (
                [],
                np.zeros((f_num, 2, 0), dtype=np.float32),
                np.zeros((f_num, 0), dtype=bool),
            )

        doc_ids = sorted(union.keys())
        d_num = len(doc_ids)
        scores = np.zeros((f_num, 2, d_num), dtype=np.float32)
        mask = np.zeros((f_num, d_num), dtype=bool)
        for di, doc_id in enumerate(doc_ids):
            scores[:, :, di] = union[doc_id]
            col = self.id_to_col[doc_id]
            mask[:, di] = self.field_mask[:, col]
        return doc_ids, scores, mask

    def rank_single_hybrid(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
    ) -> list[str]:
        """MFAR2-style single-field hybrid baseline."""
        union: dict[str, float] = {}
        for item in self._single_sparse.search(query_text, k).items:
            union[item.doc_id] = union.get(item.doc_id, 0.0) + item.score
        for item in self._single_dense.search(query_text, k).items:
            union[item.doc_id] = union.get(item.doc_id, 0.0) + item.score
        return sorted(union, key=lambda d: -union[d])[:k]

    def close(self) -> None:
        self._tmpdir.cleanup()


def example_from_shortlist_mag(
    query: MagQuery,
    doc_ids: list[str],
    scores: np.ndarray,
    mask: np.ndarray,
) -> StarkRankingExample:
    """Build a StarkRankingExample from a MAG query and its shortlist."""
    rel = np.zeros(len(doc_ids), dtype=np.float32)
    ans = {str(a) for a in query.answer_ids}
    for i, doc_id in enumerate(doc_ids):
        if doc_id in ans:
            rel[i] = 1.0
    return StarkRankingExample(
        query_text=query.query,
        doc_ids=doc_ids,
        scores=scores,
        relevance=rel,
        field_mask=mask,
    )
