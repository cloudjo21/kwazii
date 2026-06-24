"""Unified disk-backed field index store for STaRK-Prime and STaRK-MAG.

Supports two on-disk layouts without an encoder at query time — query
embeddings are pre-computed by the caller:

  MAG:   doc_ids.npy / field_mask.npy / {field}/dense.faiss + {field}/sparse/
  Prime: doc_ids.txt / field_mask.u8.mmap / {slug}/{dense.faiss or dense.f32.mmap}
           + {slug}/sparse/

Use ``is_disk_index_built`` to check readiness, then construct
``StarkDiskIndexStore`` for querying.
"""

import asyncio
import json
import logging
import os
import resource
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import marisa_trie
import numpy as np

from asmr import vocab
from asmr.index import bm25

logger = logging.getLogger(__name__)

_MANIFEST_FILE = "manifest.json"
_VOCAB_FILE = "vocab.trie"
_DOC_ID_MAPPING_FILE = "doc_id_mapping.marisa"
_DENSE_FAISS_FILE = "dense.faiss"
_DENSE_MEMMAP_FILE = "dense.f32.mmap"
_FIELD_MASK_NPY = "field_mask.npy"
_FIELD_MASK_MMAP = "field_mask.u8.mmap"
_DOC_IDS_NPY = "doc_ids.npy"
_DOC_IDS_TXT = "doc_ids.txt"
_SPARSE_DIR = "sparse"

_DENSE_BACKEND_FAISS = "faiss_flat_ip"
_DENSE_BACKEND_MEMMAP = "memmap"

_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _run_async(coro: object) -> object:
    global _ASYNC_LOOP
    if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
        _ASYNC_LOOP = asyncio.new_event_loop()
    return _ASYNC_LOOP.run_until_complete(coro)  # type: ignore[arg-type]


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _field_slug(field_name: str) -> str:
    return field_name.replace(" ", "_")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _dense_topk_faiss(
    index: faiss.Index,
    doc_ids: list[str],
    query_emb: np.ndarray,
    k: int,
) -> list[tuple[str, float]]:
    q = query_emb.astype(np.float32).reshape(1, -1).copy()
    faiss.normalize_L2(q)
    k = min(k, index.ntotal)
    if k <= 0:
        return []
    scores, ids = index.search(q, k)
    hits: list[tuple[str, float]] = []
    for idx, score in zip(ids[0], scores[0], strict=True):
        if idx < 0:
            continue
        hits.append((doc_ids[int(idx)], float(score)))
    return hits


def _dense_topk_memmap(
    vectors: np.memmap,
    doc_ids: list[str],
    query_emb: np.ndarray,
    k: int,
) -> list[tuple[str, float]]:
    q = query_emb.astype(np.float32)
    scores: np.ndarray = vectors @ q
    k = min(k, int(scores.shape[0]))
    if k <= 0:
        return []
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(doc_ids[int(i)], float(scores[int(i)])) for i in top_idx]


def _load_doc_id_mapping(path: Path) -> bm25.DocumentIndexToIdMapping:
    trie = marisa_trie.BytesTrie()
    trie.load(str(path))
    return bm25.DocumentIndexToIdMapping(trie, bm25.DocIdPostingPolicy.UNIQUE)


def _build_doc_id_mapping(doc_ids: list[str]) -> bm25.DocumentIndexToIdMapping:
    return bm25.DocumentIndexToIdMapping.build(
        enumerate(doc_ids),
        bm25.DocIdPostingPolicy.UNIQUE,
    )


def _load_doc_ids(index_dir: Path) -> list[str]:
    """Load doc_ids from either MAG (.npy) or Prime (.txt) format."""
    npy_path = index_dir / _DOC_IDS_NPY
    txt_path = index_dir / _DOC_IDS_TXT
    if npy_path.exists():
        return np.load(str(npy_path), allow_pickle=True).tolist()  # type: ignore[no-any-return]
    if txt_path.exists():
        return [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
    raise FileNotFoundError(f"No doc_ids file found in {index_dir}")


def _load_field_mask(
    index_dir: Path,
    num_fields: int,
    num_docs: int,
) -> np.ndarray:
    """Load field_mask from either MAG (.npy) or Prime (.u8.mmap) format."""
    npy_path = index_dir / _FIELD_MASK_NPY
    mmap_path = index_dir / _FIELD_MASK_MMAP
    if npy_path.exists():
        return np.load(str(npy_path))  # type: ignore[no-any-return]
    if mmap_path.exists():
        mm = np.memmap(
            mmap_path,
            dtype=np.uint8,
            mode="r",
            shape=(num_fields, num_docs),
        )
        return mm.astype(bool)
    raise FileNotFoundError(f"No field_mask file found in {index_dir}")


def _load_sparse(
    field_name: str,
    sparse_dir: Path,
    field_dir: Path,
    doc_ids: list[str],
) -> Optional[bm25.BM25Index]:
    """Load BM25 sparse index, attaching doc_id_mapping for search_topk."""
    vocab_path = sparse_dir / _VOCAB_FILE
    if not vocab_path.exists():
        return None
    trie: marisa_trie.Trie = marisa_trie.Trie()
    trie.load(str(vocab_path))
    vocabulary = vocab.Vocabulary(trie)
    index: bm25.BM25Index = bm25.BM25Indexer.load(
        field_name, vocabulary, index_dir=sparse_dir
    )
    mapping_path = field_dir / _DOC_ID_MAPPING_FILE
    if mapping_path.exists():
        index.doc_id_mapping = _load_doc_id_mapping(mapping_path)
    else:
        index.doc_id_mapping = _build_doc_id_mapping(doc_ids)
    return index


@dataclass
class ShortlistTiming:
    """Per-query shortlist stage timings in milliseconds."""

    sparse_ms: float = 0.0
    dense_ms: float = 0.0
    total_ms: float = 0.0
    field_timings: list[dict[str, float | str]] = field(default_factory=list)


class _FieldCache:
    """Process-lifetime lazy cache for per-field sparse + dense indexes."""

    def __init__(
        self,
        index_dir: Path,
        field_names: list[str],
        doc_ids: list[str],
        *,
        dense_backend: str,
        num_docs: int,
        embedding_dim: int,
    ) -> None:
        self._index_dir = index_dir
        self._field_names = field_names
        self._doc_ids = doc_ids
        self._dense_backend = dense_backend
        self._num_docs = num_docs
        self._embedding_dim = embedding_dim
        self._sparse: dict[str, Optional[bm25.BM25Index]] = {}
        self._faiss_cache: dict[str, faiss.Index] = {}
        self._mmap: dict[str, np.memmap] = {}
        self._lock = threading.Lock()

    def _field_dir(self, field: str) -> Path:
        return self._index_dir / _field_slug(field)

    def get_sparse(self, field: str) -> Optional[bm25.BM25Index]:
        with self._lock:
            if field not in self._sparse:
                fd = self._field_dir(field)
                self._sparse[field] = _load_sparse(
                    field, fd / _SPARSE_DIR, fd, self._doc_ids
                )
        return self._sparse[field]

    def get_dense_faiss(self, field: str) -> faiss.Index:
        with self._lock:
            if field not in self._faiss_cache:
                path = self._field_dir(field) / _DENSE_FAISS_FILE
                if not path.exists():
                    raise FileNotFoundError(f"Missing FAISS index: {path}")
                self._faiss_cache[field] = faiss.read_index(str(path))
        return self._faiss_cache[field]

    def get_dense_memmap(self, field: str) -> np.memmap:
        with self._lock:
            if field not in self._mmap:
                path = self._field_dir(field) / _DENSE_MEMMAP_FILE
                self._mmap[field] = np.memmap(
                    path,
                    dtype=np.float32,
                    mode="r",
                    shape=(self._num_docs, self._embedding_dim),
                )
        return self._mmap[field]


class StarkDiskIndexStore:
    """Unified disk-backed field index store for STaRK-Prime and STaRK-MAG.

    Works with both on-disk layouts — no encoder required at query time.
    Call ``build_mag_disk_index`` or ``build_prime_disk_index`` first, then
    construct this class and call ``shortlist_hybrid`` or
    ``shortlist_hybrid_dispatch``.
    """

    def __init__(self, index_dir: Path) -> None:
        manifest_path = index_dir / _MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing index manifest at {index_dir}")
        manifest: dict[str, object] = json.loads(manifest_path.read_text())

        raw_fields = manifest.get("fields", [])
        self.field_names: list[str] = (
            list(raw_fields) if isinstance(raw_fields, list) else []
        )

        raw_dim = manifest.get("embedding_dim", 0)
        self._embedding_dim: int = (
            int(raw_dim) if isinstance(raw_dim, (int, float)) else 0
        )

        raw_backend = manifest.get("dense_backend")
        self._dense_backend: str = (
            raw_backend if isinstance(raw_backend, str) else _DENSE_BACKEND_FAISS
        )

        self.doc_ids: list[str] = _load_doc_ids(index_dir)
        self.num_docs: int = len(self.doc_ids)
        self.id_to_col: dict[str, int] = {d: i for i, d in enumerate(self.doc_ids)}
        self.field_mask: np.ndarray = _load_field_mask(
            index_dir, len(self.field_names), self.num_docs
        )
        self._cache = _FieldCache(
            index_dir,
            self.field_names,
            self.doc_ids,
            dense_backend=self._dense_backend,
            num_docs=self.num_docs,
            embedding_dim=self._embedding_dim,
        )

    def _dense_topk(
        self,
        field: str,
        query_emb: np.ndarray,
        k: int,
    ) -> list[tuple[str, float]]:
        if self._dense_backend == _DENSE_BACKEND_MEMMAP:
            return _dense_topk_memmap(
                self._cache.get_dense_memmap(field), self.doc_ids, query_emb, k
            )
        return _dense_topk_faiss(
            self._cache.get_dense_faiss(field), self.doc_ids, query_emb, k
        )

    def _shortlist_one_field(
        self,
        fi: int,
        fname: str,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
        f_num: int,
        q_tok: list[str],
    ) -> tuple[dict[str, np.ndarray], float, float]:
        union_part: dict[str, np.ndarray] = {}
        sparse_ms = 0.0
        dense_ms = 0.0

        sparse = self._cache.get_sparse(fname)
        if sparse is not None:
            t0 = time.perf_counter()
            for doc_id, score in sparse.search_topk(q_tok, k):
                union_part.setdefault(doc_id, np.zeros((f_num, 2), dtype=np.float32))
                union_part[doc_id][fi, 0] = max(union_part[doc_id][fi, 0], score)
            sparse_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for doc_id, score in self._dense_topk(fname, query_emb, k):
            union_part.setdefault(doc_id, np.zeros((f_num, 2), dtype=np.float32))
            union_part[doc_id][fi, 1] = max(union_part[doc_id][fi, 1], score)
        dense_ms = (time.perf_counter() - t0) * 1000.0

        return union_part, sparse_ms, dense_ms

    def _merge_union(
        self,
        parts: list[dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        merged: dict[str, np.ndarray] = {}
        for part in parts:
            for doc_id, scores in part.items():
                if doc_id not in merged:
                    merged[doc_id] = scores.copy()
                else:
                    merged[doc_id] = np.maximum(merged[doc_id], scores)
        return merged

    def _union_to_tensors(
        self,
        union: dict[str, np.ndarray],
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        f_num = len(self.field_names)
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

    def shortlist_hybrid(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
        *,
        timing: Optional[ShortlistTiming] = None,
        log_timing: bool = False,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Union per-field lex+dense top-k into [F, 2, D] scores."""
        t_total = time.perf_counter()
        f_num = len(self.field_names)
        union: dict[str, np.ndarray] = {}
        q_tok = _tokenize(query_text)
        total_sparse_ms = 0.0
        total_dense_ms = 0.0

        for fi, fname in enumerate(self.field_names):
            part, sparse_ms, dense_ms = self._shortlist_one_field(
                fi, fname, query_text, query_emb, k, f_num, q_tok
            )
            for doc_id, scores in part.items():
                if doc_id not in union:
                    union[doc_id] = scores
                else:
                    union[doc_id] = np.maximum(union[doc_id], scores)
            total_sparse_ms += sparse_ms
            total_dense_ms += dense_ms
            if timing is not None:
                timing.field_timings.append(
                    {"field": fname, "sparse_ms": sparse_ms, "dense_ms": dense_ms}
                )

        if timing is not None:
            timing.sparse_ms = total_sparse_ms
            timing.dense_ms = total_dense_ms
            timing.total_ms = (time.perf_counter() - t_total) * 1000.0

        if log_timing:
            logger.info(
                "shortlist sparse_ms=%.1f dense_ms=%.1f rss_mb=%.1f",
                total_sparse_ms,
                total_dense_ms,
                _rss_mb(),
            )

        return self._union_to_tensors(union)

    async def _shortlist_hybrid_async(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
        *,
        timing: Optional[ShortlistTiming] = None,
        log_timing: bool = False,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        t_total = time.perf_counter()
        f_num = len(self.field_names)
        q_tok = _tokenize(query_text)

        async def _one(
            fi: int, fname: str
        ) -> tuple[dict[str, np.ndarray], float, float, str]:
            part, sp, dn = await asyncio.to_thread(
                self._shortlist_one_field,
                fi,
                fname,
                query_text,
                query_emb,
                k,
                f_num,
                q_tok,
            )
            return part, sp, dn, fname

        results = await asyncio.gather(
            *[_one(fi, fname) for fi, fname in enumerate(self.field_names)]
        )
        parts = [r[0] for r in results]
        union = self._merge_union(parts)
        total_sparse_ms = sum(r[1] for r in results)
        total_dense_ms = sum(r[2] for r in results)

        if timing is not None:
            timing.sparse_ms = total_sparse_ms
            timing.dense_ms = total_dense_ms
            timing.total_ms = (time.perf_counter() - t_total) * 1000.0
            timing.field_timings = [
                {"field": r[3], "sparse_ms": r[1], "dense_ms": r[2]} for r in results
            ]

        if log_timing:
            logger.info(
                "parallel shortlist sparse_ms=%.1f dense_ms=%.1f total_ms=%.1f rss_mb=%.1f",
                total_sparse_ms,
                total_dense_ms,
                (time.perf_counter() - t_total) * 1000.0,
                _rss_mb(),
            )

        return self._union_to_tensors(union)

    def shortlist_hybrid_dispatch(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
        *,
        parallel: bool = False,
        timing: Optional[ShortlistTiming] = None,
        log_timing: bool = False,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Serial or parallel per-field hybrid shortlist."""
        if parallel:
            return _run_async(  # type: ignore[return-value]
                self._shortlist_hybrid_async(
                    query_text,
                    query_emb,
                    k,
                    timing=timing,
                    log_timing=log_timing,
                )
            )
        return self.shortlist_hybrid(
            query_text, query_emb, k, timing=timing, log_timing=log_timing
        )

    def rank_single_hybrid(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
    ) -> list[str]:
        """Single concatenated-field hybrid ranking (MAG-style)."""
        union: dict[str, float] = {}
        sparse = self._cache.get_sparse("single")
        if sparse is not None:
            q_tok = _tokenize(query_text)
            for doc_id, score in sparse.search_topk(q_tok, k):
                union[doc_id] = union.get(doc_id, 0.0) + score
        for doc_id, score in self._dense_topk("single", query_emb, k):
            union[doc_id] = union.get(doc_id, 0.0) + score
        return sorted(union, key=lambda d: -union[d])[:k]

    def close(self) -> None:
        pass


def is_disk_index_built(index_dir: Path) -> bool:
    """Return True when a valid index exists in index_dir (either format)."""
    if not (index_dir / _MANIFEST_FILE).exists():
        return False
    return (index_dir / _DOC_IDS_NPY).exists() or (index_dir / _DOC_IDS_TXT).exists()
