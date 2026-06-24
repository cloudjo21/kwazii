"""Disk-persistent field indexes for STaRK-MAG (stark_prime parity).

Layout::

    {index_dir}/
      manifest.json          # MagIndexManifest (Pydantic)
      field_mask.npy         # bool [F, D]
      doc_ids.npy            # str [D]
      {field_slug}/
        dense.faiss          # faiss.IndexFlatIP
        dense.faiss.doc_ids.npy
        sparse/              # BM25 CSC files + vocab.trie
      single/
        dense.faiss
        dense.faiss.doc_ids.npy
        sparse/
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import marisa_trie
import numpy as np
from pydantic import BaseModel, ConfigDict

from asmr import columnar, vocab
from asmr.datasets.stark_mag.loader import (
    MAG_FIELD_NAMES,
    MagCorpus,
    field_text,
    single_field_text,
)
from asmr.encode.protocol import TextEncoderProtocol
from asmr.index import bm25
from asmr.index.config import FieldConfig, IndexUsage, RepresentationType, TokenizerType
from asmr.index.fields import SparseTextFieldIndex
from asmr.index.text_encoder import TextEncodingIndexer
from asmr.tokenize.helpers import TokenizerWrapper

logger = logging.getLogger(__name__)

_MAG_INDEX_VERSION = 1
_MANIFEST_FILE = "manifest.json"
_VOCAB_FILE = "vocab.trie"
_DENSE_FAISS_FILE = "dense.faiss"
_FIELD_MASK_FILE = "field_mask.npy"
_DOC_IDS_FILE = "doc_ids.npy"

_ALL_FIELD_SLUGS = list(MAG_FIELD_NAMES) + ["single"]


class MagIndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    fields: list[str]
    embedding_dim: int
    complete: bool


def is_mag_index_built(index_dir: Path) -> bool:
    """Return True iff the index at index_dir exists and is marked complete."""
    path = index_dir / _MANIFEST_FILE
    if not path.exists():
        return False
    try:
        m = MagIndexManifest.model_validate_json(path.read_text())
        return bool(m.complete)
    except Exception:
        return False


def _dense_cfg(fname: str) -> FieldConfig:
    return FieldConfig(
        fname,
        TokenizerType.SPLIT,
        RepresentationType.DENSE,
        usage=IndexUsage.BENCHMARK,
    )


def _sparse_cfg(fname: str) -> FieldConfig:
    return FieldConfig(fname, TokenizerType.SPLIT, RepresentationType.SPARSE)


def _build_field_sparse(
    field_name: str,
    texts: list[str],
    doc_ids: list[str],
    sparse_dir: Path,
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = TokenizerWrapper.from_config(_sparse_cfg(field_name))
    all_tokens: set[str] = set()
    for text in texts:
        all_tokens.update(tokenizer.tokenize(text))

    vocabulary = vocab.Vocabulary.from_token_set(all_tokens)
    field_columnar = columnar.FieldBasedColumnarTexts(
        [{field_name: text} for text in texts],
        field_name,
        len(texts),
    )
    field_stats = columnar.ColumnarStatisticsBuilder.build(field_columnar, vocabulary)
    bm25.BM25Indexer.build(
        field_name=field_name,
        columnar_posting=texts,
        vocab=vocabulary,
        field_statistics=field_stats,
        doc_ids=doc_ids,
        index_dir=sparse_dir,
    )
    vocabulary.trie.save(str(sparse_dir / _VOCAB_FILE))


def _load_field_sparse(
    field_name: str, sparse_dir: Path
) -> Optional[SparseTextFieldIndex]:
    vocab_path = sparse_dir / _VOCAB_FILE
    if not vocab_path.exists():
        return None
    trie = marisa_trie.Trie()
    trie.load(str(vocab_path))
    vocabulary = vocab.Vocabulary(trie)
    bm25_index = bm25.BM25Indexer.load(field_name, vocabulary, index_dir=sparse_dir)
    return SparseTextFieldIndex(_sparse_cfg(field_name), index=bm25_index)


def build_mag_disk_index(
    corpus: MagCorpus,
    encoder: TextEncoderProtocol,
    index_dir: Path,
    batch_size: int = 64,
    *,
    rebuild: bool = False,
) -> None:
    """Build and persist per-field BM25 + FAISS indexes for STaRK-MAG.

    Skips build if the index already exists and is marked complete, unless
    rebuild=True.
    """
    if not rebuild and is_mag_index_built(index_dir):
        logger.info("MAG disk index cache hit — skipping build at %s", index_dir)
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = corpus.doc_ids
    field_names = list(MAG_FIELD_NAMES)

    # Build field_mask: [F, D] where F = MAG_FIELD_NAMES (no "single")
    field_mask = np.zeros((len(field_names), len(doc_ids)), dtype=bool)
    for fi, fname in enumerate(field_names):
        texts = [field_text(doc, fname) for doc in corpus.documents]
        field_mask[fi] = np.array([bool(t.strip()) for t in texts], dtype=bool)
    np.save(str(index_dir / _FIELD_MASK_FILE), field_mask)
    np.save(str(index_dir / _DOC_IDS_FILE), np.array(doc_ids, dtype=object))

    for fname in _ALL_FIELD_SLUGS:
        field_dir = index_dir / fname
        field_dir.mkdir(exist_ok=True)
        logger.info("Building MAG disk index field: %s", fname)

        if fname == "single":
            texts = [single_field_text(doc) for doc in corpus.documents]
        else:
            texts = [field_text(doc, fname) for doc in corpus.documents]

        # Placeholder text for empty fields so indexes stay aligned
        nonempty = [bool(t.strip()) for t in texts]
        padded = [t if t.strip() else fname for t in texts]

        if any(nonempty):
            _build_field_sparse(fname, padded, doc_ids, field_dir / "sparse")
            logger.info("  sparse done: %s", fname)

        dense_indexer = TextEncodingIndexer(encoder, _dense_cfg(fname))
        dense_indexer.add_documents(doc_ids, padded)
        dense_indexer.save_index(str(field_dir / _DENSE_FAISS_FILE))
        logger.info("  dense done: %s", fname)

    manifest = MagIndexManifest(
        version=_MAG_INDEX_VERSION,
        fields=field_names,
        embedding_dim=encoder.embedding_dim,
        complete=True,
    )
    (index_dir / _MANIFEST_FILE).write_text(manifest.model_dump_json())
    logger.info("MAG disk index complete at %s", index_dir)


class MagFieldIndexCache:
    """Process-lifetime warm cache for loaded MAG per-field indexes."""

    def __init__(self, index_dir: Path, encoder: TextEncoderProtocol) -> None:
        self._index_dir = index_dir
        self._encoder = encoder
        self._sparse: dict[str, Optional[SparseTextFieldIndex]] = {}
        self._dense: dict[str, TextEncodingIndexer] = {}
        self._lock = threading.Lock()

    def get_sparse(self, fname: str) -> Optional[SparseTextFieldIndex]:
        with self._lock:
            if fname not in self._sparse:
                self._sparse[fname] = _load_field_sparse(
                    fname, self._index_dir / fname / "sparse"
                )
        return self._sparse[fname]

    def get_dense(self, fname: str) -> TextEncodingIndexer:
        with self._lock:
            if fname not in self._dense:
                indexer = TextEncodingIndexer(self._encoder, _dense_cfg(fname))
                indexer.load_index(str(self._index_dir / fname / _DENSE_FAISS_FILE))
                self._dense[fname] = indexer
        return self._dense[fname]


class MagDiskIndexStore:
    """Disk-backed MAG field indexes — same public API as MagFieldIndexes.

    Use build_mag_disk_index() to populate the store, then construct this
    class to query it. Indexes are loaded lazily and cached for the process
    lifetime.
    """

    def __init__(self, index_dir: Path, encoder: TextEncoderProtocol) -> None:
        manifest = MagIndexManifest.model_validate_json(
            (index_dir / _MANIFEST_FILE).read_text()
        )
        self.field_names = manifest.fields
        self.doc_ids: list[str] = np.load(
            str(index_dir / _DOC_IDS_FILE), allow_pickle=True
        ).tolist()
        self.num_docs = len(self.doc_ids)
        self.id_to_col = {d: i for i, d in enumerate(self.doc_ids)}
        self.field_mask: np.ndarray = np.load(str(index_dir / _FIELD_MASK_FILE))
        self._cache = MagFieldIndexCache(index_dir, encoder)

    def shortlist_hybrid(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """MFARAll shortlist: union of per-field top-k (lex + dense).

        Same signature as MagFieldIndexes.shortlist_hybrid().
        """
        f_num = len(self.field_names)
        union: dict[str, np.ndarray] = {}

        for fi, fname in enumerate(self.field_names):
            sparse = self._cache.get_sparse(fname)
            if sparse is not None:
                for item in sparse.search(query_text, k).items:
                    union.setdefault(
                        item.doc_id, np.zeros((f_num, 2), dtype=np.float32)
                    )
                    union[item.doc_id][fi, 0] = max(
                        union[item.doc_id][fi, 0], item.score
                    )
            dense = self._cache.get_dense(fname)
            for item in dense.search(query_text, k).items:
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
        sparse = self._cache.get_sparse("single")
        if sparse is not None:
            for item in sparse.search(query_text, k).items:
                union[item.doc_id] = union.get(item.doc_id, 0.0) + item.score
        dense = self._cache.get_dense("single")
        for item in dense.search(query_text, k).items:
            union[item.doc_id] = union.get(item.doc_id, 0.0) + item.score
        return sorted(union, key=lambda d: -union[d])[:k]

    def close(self) -> None:
        pass
