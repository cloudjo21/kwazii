"""STaRK-Prime disk index v2 — asmr module integration + QueryRouter shortlist."""

import argparse
import asyncio
import json
import logging
import os
import resource
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import faiss
import marisa_trie
import numpy as np
import torch

from asmr import columnar
from asmr import vocab
from asmr.datasets.stark_prime.loader import (
    PRIME_FIELD_NAMES,
    PrimeCorpus,
    build_prime_corpus,
    field_text,
)
from asmr.evaluation.jina_gpu_oom import (
    JINA_OOM_ENCODE_BATCH,
    clear_cuda_cache,
    is_cuda_oom,
)
from asmr.evaluation.query_encoders import QueryEncoderProtocol, create_query_encoder
from asmr.evaluation.stark_prime_router import shortlist_prime_hybrid_router
from asmr.index import bm25
from asmr.tokenize import helpers
from asmr.train.query_encoder import HfQueryEncoder
from fde.config import fde_config_fingerprint

logger = logging.getLogger(__name__)

_MANIFEST = "manifest.json"
_DOC_IDS = "doc_ids.txt"
_FIELD_MASK = "field_mask.u8.mmap"
_DENSE_FILE = "dense.f32.mmap"
_DENSE_FAISS = "dense.faiss"
_DOC_ID_MAPPING = "doc_id_mapping.marisa"
_SPARSE_DIR = "sparse"
_VOCAB_FILE = "vocab.trie"

_DENSE_BACKEND_MEMMAP = "memmap"
_DENSE_BACKEND_FAISS = "faiss_flat_ip"

_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None


def _run_async(coro):  # type: ignore[no-untyped-def]
    """Run one coroutine on a reused loop (avoids asyncio.run per-query leaks)."""
    global _ASYNC_LOOP
    if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
        _ASYNC_LOOP = asyncio.new_event_loop()
    return _ASYNC_LOOP.run_until_complete(coro)


def _rss_mb() -> float:
    """Peak RSS of this process in MB (Linux: ru_maxrss is KiB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _field_slug(field_name: str) -> str:
    return field_name.replace(" ", "_")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@contextmanager
def _bm25_index_dir(path: Path) -> Iterator[None]:
    old = bm25._INDEX_DIR
    bm25._INDEX_DIR = str(path)
    try:
        yield
    finally:
        bm25._INDEX_DIR = old


def _save_doc_id_mapping(
    mapping: bm25.DocumentIndexToIdMapping,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping.mapping.save(str(path))


def _load_doc_id_mapping(path: Path) -> bm25.DocumentIndexToIdMapping:
    trie = marisa_trie.BytesTrie()
    trie.load(str(path))
    return bm25.DocumentIndexToIdMapping(
        trie,
        bm25.DocIdPostingPolicy.UNIQUE,
    )


def _build_doc_id_mapping(doc_ids: list[str]) -> bm25.DocumentIndexToIdMapping:
    return bm25.DocumentIndexToIdMapping.build(
        enumerate(doc_ids),
        bm25.DocIdPostingPolicy.UNIQUE,
    )


def _sparse_topk(
    index: bm25.BM25Index,
    terms: list[str],
    k: int,
) -> list[tuple[str, float]]:
    """Score only postings for query terms; return top-k (doc_id, score)."""
    return index.search_topk(terms, k)


def _dense_topk_memmap(
    vectors: np.memmap,
    doc_ids: list[str],
    query_emb: np.ndarray,
    k: int,
) -> list[tuple[str, float]]:
    """Top-k by dot product against memmap rows."""
    q = query_emb.astype(np.float32)
    scores = vectors @ q
    k = min(k, scores.shape[0])
    if k <= 0:
        return []
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(doc_ids[int(i)], float(scores[int(i)])) for i in top_idx]


def _dense_topk_faiss(
    index: faiss.Index,
    doc_ids: list[str],
    query_emb: np.ndarray,
    k: int,
) -> list[tuple[str, float]]:
    """Top-k by inner product via FAISS IndexFlatIP."""
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


def export_corpus_jsonl(corpus: PrimeCorpus, path: Path) -> None:
    """Write reusable corpus JSONL (doc_id + mFAR document)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc_id, document in zip(corpus.doc_ids, corpus.documents):
            row = {"doc_id": doc_id, "document": document}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_doc_ids(doc_ids: list[str], path: Path) -> None:
    path.write_text("\n".join(doc_ids) + "\n", encoding="utf-8")


def _read_doc_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _build_sparse_field(
    field_name: str,
    texts: list[str],
    doc_ids: list[str],
    out_dir: Path,
    *,
    field_dir: Path,
) -> None:
    """Build CSC BM25 index for one field on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = helpers.SplitTokenizer()

    all_tokens: set[str] = set()
    for text in texts:
        all_tokens.update(tokenizer.tokenize(text))

    vocabulary = vocab.Vocabulary.from_token_set(all_tokens)
    field_columnar = columnar.FieldBasedColumnarTexts(
        [{field_name: text} for text in texts],
        field_name,
        len(texts),
    )
    field_stats = columnar.ColumnarStatisticsBuilder.build(
        field_columnar,
        vocabulary,
    )

    with _bm25_index_dir(out_dir):
        bm25.BM25Indexer.build(
            field_name=field_name,
            columnar_posting=texts,
            vocab=vocabulary,
            field_statistics=field_stats,
            doc_ids=doc_ids,
            index_dir=out_dir,
        )

    vocabulary.trie.save(str(out_dir / _VOCAB_FILE))
    mapping = _build_doc_id_mapping(doc_ids)
    _save_doc_id_mapping(mapping, field_dir / _DOC_ID_MAPPING)


def _load_sparse_field(
    field_name: str,
    sparse_dir: Path,
    *,
    field_dir: Path,
    index_dir: Path,
) -> bm25.BM25Index:
    vocab_path = sparse_dir / _VOCAB_FILE
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing sparse vocab: {vocab_path}")
    trie = marisa_trie.Trie()
    trie.load(str(vocab_path))
    vocabulary = vocab.Vocabulary(trie)

    with _bm25_index_dir(sparse_dir):
        index = bm25.BM25Indexer.load(
            field_name,
            vocabulary,
            index_dir=sparse_dir,
        )

    mapping_path = field_dir / _DOC_ID_MAPPING
    if mapping_path.exists():
        index.doc_id_mapping = _load_doc_id_mapping(mapping_path)
    else:
        doc_ids_path = index_dir / _DOC_IDS
        doc_ids = _read_doc_ids(doc_ids_path)
        index.doc_id_mapping = _build_doc_id_mapping(doc_ids)
    return index


def _recover_jina_gpu_after_oom(encoder: object, out_path: Path) -> None:
    """Reset partial dense artifact and apply Jina OOM mitigations."""
    if out_path.exists():
        out_path.unlink()
    clear_cuda_cache()
    setter = getattr(encoder, "set_jina_encode_batch_size", None)
    if setter is not None:
        setter(JINA_OOM_ENCODE_BATCH)
    release = getattr(encoder, "release_gpu", None)
    if release is not None:
        release()
    ensure = getattr(encoder, "ensure_gpu", None)
    if ensure is not None:
        ensure()
    enable = getattr(encoder, "enable_fieldwise_gpu", None)
    if enable is not None:
        enable()


def _build_dense_field_impl(
    texts: list[str],
    encoder: HfQueryEncoder,
    out_path: Path,
    *,
    batch_size: int = 64,
) -> tuple[int, int]:
    """Encode field texts to memmap; return (n_docs, dim)."""
    n_docs = len(texts)
    if n_docs == 0:
        return 0, encoder.embedding_dim

    sample = encoder.encode(texts[:1], batch_size=1)
    dim = int(sample.shape[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(
        out_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_docs, dim),
    )

    row = 0
    for start in range(0, n_docs, batch_size):
        batch = texts[start : start + batch_size]
        emb = encoder.encode(batch, batch_size=batch_size).numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        mm[row : row + len(batch)] = (emb / np.maximum(norms, 1e-12)).astype(np.float32)
        row += len(batch)

    mm.flush()
    del mm
    return n_docs, dim


def _build_dense_field(
    texts: list[str],
    encoder: HfQueryEncoder,
    out_path: Path,
    *,
    batch_size: int = 64,
) -> tuple[int, int]:
    """Encode field texts with optional Jina CUDA OOM recovery."""
    ensure = getattr(encoder, "ensure_gpu", None)
    if ensure is not None:
        ensure()
    try:
        return _build_dense_field_impl(
            texts,
            encoder,
            out_path,
            batch_size=batch_size,
        )
    except Exception as exc:
        if not is_cuda_oom(exc) or getattr(encoder, "release_gpu", None) is None:
            raise
        logger.warning(
            "CUDA OOM building dense field at %s; retry after GPU recovery",
            out_path,
        )
        _recover_jina_gpu_after_oom(encoder, out_path)
        return _build_dense_field_impl(
            texts,
            encoder,
            out_path,
            batch_size=batch_size,
        )


def build_dense_faiss_field(
    memmap_path: Path,
    faiss_path: Path,
    *,
    n_docs: int,
    dim: int,
    batch_size: int = 10000,
) -> None:
    """Build IndexFlatIP from L2-normalized memmap vectors."""
    mm = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="r",
        shape=(n_docs, dim),
    )
    index = faiss.IndexFlatIP(dim)
    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        batch = np.array(mm[start:end], copy=True)
        faiss.normalize_L2(batch)
        index.add(batch)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    del mm


def _manifest_dense_backend(manifest: dict[str, object]) -> str:
    backend = manifest.get("dense_backend")
    if isinstance(backend, str):
        return backend
    raw_version = manifest.get("version", 2)
    version = int(raw_version) if isinstance(raw_version, (int, str)) else 2
    if version >= 3:
        return _DENSE_BACKEND_FAISS
    return _DENSE_BACKEND_MEMMAP


def is_index_built(index_dir: Path) -> bool:
    """Return True when manifest and all per-field artifacts exist."""
    manifest_path = index_dir / _MANIFEST
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    dense_backend = _manifest_dense_backend(manifest)
    for fname in manifest.get("fields", []):
        slug = _field_slug(fname)
        field_dir = index_dir / slug
        sparse = field_dir / _SPARSE_DIR / "metadata.json"
        if not sparse.exists():
            return False
        if dense_backend == _DENSE_BACKEND_FAISS:
            dense = field_dir / _DENSE_FAISS
        else:
            dense = field_dir / _DENSE_FILE
        if not dense.exists():
            return False
    return True


@dataclass
class ShortlistTiming:
    """Per-query shortlist stage timings in milliseconds."""

    sparse_ms: float = 0.0
    dense_ms: float = 0.0
    total_ms: float = 0.0
    field_timings: list[dict[str, float | str]] = field(default_factory=list)


class FieldIndexCache:
    """Process-lifetime warm cache for per-field sparse + dense indexes."""

    def __init__(
        self,
        index_dir: Path,
        field_names: list[str],
        *,
        num_docs: int,
        embedding_dim: int,
        dense_backend: str,
    ) -> None:
        self.index_dir = index_dir
        self.field_names = field_names
        self.num_docs = num_docs
        self.embedding_dim = embedding_dim
        self.dense_backend = dense_backend
        self._sparse: dict[str, bm25.BM25Index] = {}
        self._faiss: dict[str, faiss.Index] = {}
        self._dense_mmaps: dict[str, np.memmap] = {}
        self._lock = threading.Lock()

    def get_sparse(self, field: str) -> bm25.BM25Index:
        with self._lock:
            if field not in self._sparse:
                slug = _field_slug(field)
                field_dir = self.index_dir / slug
                sparse_dir = field_dir / _SPARSE_DIR
                self._sparse[field] = _load_sparse_field(
                    field,
                    sparse_dir,
                    field_dir=field_dir,
                    index_dir=self.index_dir,
                )
            return self._sparse[field]

    def get_dense_faiss(self, field: str) -> faiss.Index:
        with self._lock:
            if field not in self._faiss:
                slug = _field_slug(field)
                path = self.index_dir / slug / _DENSE_FAISS
                if not path.exists():
                    raise FileNotFoundError(f"Missing FAISS index: {path}")
                self._faiss[field] = faiss.read_index(str(path))
            return self._faiss[field]

    def get_dense_memmap(self, field: str) -> np.memmap:
        with self._lock:
            if field not in self._dense_mmaps:
                slug = _field_slug(field)
                path = self.index_dir / slug / _DENSE_FILE
                self._dense_mmaps[field] = np.memmap(
                    path,
                    dtype=np.float32,
                    mode="r",
                    shape=(self.num_docs, self.embedding_dim),
                )
            return self._dense_mmaps[field]


class _HfEncoderShim:
    """Adapt QueryEncoderProtocol to legacy index-build encode() shim."""

    def __init__(self, encoder: QueryEncoderProtocol) -> None:
        self._encoder = encoder

    @property
    def embedding_dim(self) -> int:
        return self._encoder.embedding_dim

    @property
    def jina_fieldwise_gpu(self) -> bool:
        return bool(getattr(self._encoder, "jina_fieldwise_gpu", False))

    def release_gpu(self) -> None:
        release = getattr(self._encoder, "release_gpu", None)
        if release is not None:
            release()

    def ensure_gpu(self) -> None:
        ensure = getattr(self._encoder, "ensure_gpu", None)
        if ensure is not None:
            ensure()

    def set_jina_encode_batch_size(self, batch_size: int) -> None:
        setter = getattr(self._encoder, "set_jina_encode_batch_size", None)
        if setter is not None:
            setter(batch_size)

    def enable_fieldwise_gpu(self) -> None:
        enable = getattr(self._encoder, "enable_fieldwise_gpu", None)
        if enable is not None:
            enable()

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        from fde.config import PromptType

        return torch.from_numpy(
            self._encoder.encode_text(
                texts,
                PromptType.PASSAGE,
                batch_size=batch_size,
            ),
        )


class PrimeDiskIndexStore:
    """Disk-backed store with QueryRouter hybrid shortlist."""

    def __init__(self, index_dir: Path) -> None:
        if not is_index_built(index_dir):
            raise FileNotFoundError(f"Disk index not built: {index_dir}")
        self.index_dir = index_dir
        manifest = json.loads((index_dir / _MANIFEST).read_text())
        self.field_names: list[str] = list(manifest["fields"])
        self.embedding_dim = int(manifest["embedding_dim"])
        self.dense_backend = _manifest_dense_backend(manifest)
        self.doc_ids = _read_doc_ids(index_dir / _DOC_IDS)
        self.id_to_col = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.num_docs = len(self.doc_ids)
        mask_mm = np.memmap(
            index_dir / _FIELD_MASK,
            dtype=np.uint8,
            mode="r",
            shape=(len(self.field_names), self.num_docs),
        )
        self.field_mask = mask_mm.astype(bool)
        self.field_cache = FieldIndexCache(
            index_dir,
            self.field_names,
            num_docs=self.num_docs,
            embedding_dim=self.embedding_dim,
            dense_backend=self.dense_backend,
        )

    def _dense_topk_field(
        self,
        field: str,
        query_emb: np.ndarray,
        k: int,
    ) -> list[tuple[str, float]]:
        if self.dense_backend == _DENSE_BACKEND_FAISS:
            index = self.field_cache.get_dense_faiss(field)
            return _dense_topk_faiss(index, self.doc_ids, query_emb, k)
        dense = self.field_cache.get_dense_memmap(field)
        return _dense_topk_memmap(dense, self.doc_ids, query_emb, k)

    def _shortlist_one_field(
        self,
        fi: int,
        fname: str,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
        *,
        f_num: int,
        q_tok: list[str],
    ) -> tuple[dict[str, np.ndarray], float, float]:
        union_part: dict[str, np.ndarray] = {}
        sparse_ms = 0.0
        dense_ms = 0.0

        slug = _field_slug(fname)
        field_dir = self.index_dir / slug
        sparse_dir = field_dir / _SPARSE_DIR
        if sparse_dir.exists():
            t0 = time.perf_counter()
            sparse = self.field_cache.get_sparse(fname)
            for doc_id, score in _sparse_topk(sparse, q_tok, k):
                union_part.setdefault(
                    doc_id,
                    np.zeros((f_num, 2), dtype=np.float32),
                )
                union_part[doc_id][fi, 0] = max(union_part[doc_id][fi, 0], score)
            sparse_ms = (time.perf_counter() - t0) * 1000.0

        dense_path = field_dir / (
            _DENSE_FAISS if self.dense_backend == _DENSE_BACKEND_FAISS else _DENSE_FILE
        )
        if dense_path.exists():
            t0 = time.perf_counter()
            for doc_id, score in self._dense_topk_field(fname, query_emb, k):
                union_part.setdefault(
                    doc_id,
                    np.zeros((f_num, 2), dtype=np.float32),
                )
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
                np.zeros(
                    (f_num, 0),
                    dtype=bool,
                ),
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
        timing: ShortlistTiming | None = None,
        log_timing: bool = False,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Union per-field lex+dense top-k into [F,2,D] scores."""
        t_total = time.perf_counter()
        f_num = len(self.field_names)
        union: dict[str, np.ndarray] = {}
        q_tok = _tokenize(query_text)
        total_sparse_ms = 0.0
        total_dense_ms = 0.0

        for fi, fname in enumerate(self.field_names):
            part, sparse_ms, dense_ms = self._shortlist_one_field(
                fi,
                fname,
                query_text,
                query_emb,
                k,
                f_num=f_num,
                q_tok=q_tok,
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
                    {
                        "field": fname,
                        "sparse_ms": sparse_ms,
                        "dense_ms": dense_ms,
                    }
                )
            if log_timing:
                logger.info(
                    "field=%s sparse_ms=%.1f dense_ms=%.1f rss_mb=%.1f",
                    fname,
                    sparse_ms,
                    dense_ms,
                    _rss_mb(),
                )

        if timing is not None:
            timing.sparse_ms = total_sparse_ms
            timing.dense_ms = total_dense_ms
            timing.total_ms = (time.perf_counter() - t_total) * 1000.0

        return self._union_to_tensors(union)

    async def shortlist_hybrid_async(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
        *,
        timing: ShortlistTiming | None = None,
        log_timing: bool = False,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Parallel per-field shortlist via asyncio.to_thread."""
        t_total = time.perf_counter()
        f_num = len(self.field_names)
        q_tok = _tokenize(query_text)

        async def _one_field(
            fi: int,
            fname: str,
        ) -> tuple[dict[str, np.ndarray], float, float, str]:
            part, sparse_ms, dense_ms = await asyncio.to_thread(
                self._shortlist_one_field,
                fi,
                fname,
                query_text,
                query_emb,
                k,
                f_num=f_num,
                q_tok=q_tok,
            )
            return part, sparse_ms, dense_ms, fname

        results = await asyncio.gather(
            *[_one_field(fi, fname) for fi, fname in enumerate(self.field_names)]
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
                "parallel shortlist sparse_ms=%.1f dense_ms=%.1f "
                "total_ms=%.1f rss_mb=%.1f",
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
        timing: ShortlistTiming | None = None,
        log_timing: bool = False,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Router-based hybrid shortlist with field_mask."""

        async def _run() -> tuple[list[str], np.ndarray, np.ndarray]:
            return await shortlist_prime_hybrid_router(
                self,
                query_text,
                query_emb,
                k,
                parallel=parallel,
                timing=timing,
            )

        doc_ids, scores, mask = _run_async(_run())
        if log_timing and timing is not None:
            logger.info(
                "router shortlist total_ms=%.1f parallel=%s",
                timing.total_ms,
                parallel,
            )
        return doc_ids, scores, mask


def build_prime_disk_index(
    data_root: Path,
    index_dir: Path,
    encoder: HfQueryEncoder,
    *,
    encoder_name: str = "facebook/contriever-msmarco",
    corpus_jsonl: Path | None = None,
    batch_size: int = 64,
    max_docs: int = -1,
    rebuild: bool = False,
    build_faiss: bool = False,
    index_metadata: dict[str, object] | None = None,
) -> None:
    """Build per-field sparse (CSC) + dense indexes on disk."""
    if is_index_built(index_dir) and not rebuild:
        logger.info("Index cache hit at %s (skip build)", index_dir)
        return

    corpus = build_prime_corpus(data_root, max_docs=max_docs)
    n_docs = len(corpus.doc_ids)
    logger.info(
        "Building disk index: %d docs, %d fields, RSS=%.1fMB",
        n_docs,
        len(PRIME_FIELD_NAMES),
        _rss_mb(),
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_jsonl or (data_root / "corpus" / "prime_corpus.jsonl")
    export_corpus_jsonl(corpus, corpus_path)
    _write_doc_ids(corpus.doc_ids, index_dir / _DOC_IDS)

    f_num = len(PRIME_FIELD_NAMES)
    field_mask = np.zeros((f_num, n_docs), dtype=np.uint8)

    for fi, fname in enumerate(PRIME_FIELD_NAMES):
        slug = _field_slug(fname)
        field_dir = index_dir / slug
        texts = [field_text(doc, fname) for doc in corpus.documents]
        for di, text in enumerate(texts):
            if text.strip():
                field_mask[fi, di] = 1

        nonempty_texts = [t if t.strip() else fname for t in texts]
        sparse_dir = field_dir / _SPARSE_DIR
        _build_sparse_field(
            fname,
            texts,
            corpus.doc_ids,
            sparse_dir,
            field_dir=field_dir,
        )

        dense_path = field_dir / _DENSE_FILE
        _build_dense_field(
            nonempty_texts,
            encoder,
            dense_path,
            batch_size=batch_size,
        )
        if build_faiss:
            build_dense_faiss_field(
                dense_path,
                field_dir / _DENSE_FAISS,
                n_docs=n_docs,
                dim=encoder.embedding_dim,
            )

        logger.info(
            "field %d/%d %s built, RSS=%.1fMB",
            fi + 1,
            f_num,
            fname,
            _rss_mb(),
        )
        if getattr(encoder, "jina_fieldwise_gpu", False):
            release = getattr(encoder, "release_gpu", None)
            if release is not None:
                logger.info(
                    "field %d/%d %s: release Jina GPU after OOM recovery",
                    fi + 1,
                    f_num,
                    fname,
                )
                release()

    mask_path = index_dir / _FIELD_MASK
    mask_mm = np.memmap(
        mask_path,
        dtype=np.uint8,
        mode="w+",
        shape=field_mask.shape,
    )
    mask_mm[:] = field_mask
    mask_mm.flush()
    del mask_mm

    try:
        corpus_rel = str(corpus_path.relative_to(data_root))
    except ValueError:
        corpus_rel = str(corpus_path)

    dense_backend = _DENSE_BACKEND_FAISS if build_faiss else _DENSE_BACKEND_MEMMAP
    manifest: dict[str, object] = {
        "version": 3 if build_faiss else 2,
        "encoder": encoder_name,
        "dense_backend": dense_backend,
        "sparse_backend": "csc_bm25",
        "embedding_dim": encoder.embedding_dim,
        "n_docs": n_docs,
        "fields": list(PRIME_FIELD_NAMES),
        "corpus_path": corpus_rel,
    }
    if build_faiss:
        manifest["features"] = [
            "field_index_cache",
            "query_emb_cache",
            "parallel_fields",
        ]
    if index_metadata:
        manifest.update(index_metadata)
    (index_dir / _MANIFEST).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote manifest %s", index_dir / _MANIFEST)


def _index_metadata_for_encoder(encoder: QueryEncoderProtocol) -> dict[str, object]:
    """Extra manifest fields for Jina FDE projection (ADR-003)."""
    fde_dim = getattr(encoder, "fde_output_dim", None)
    if fde_dim is None:
        return {}
    return {
        "fde_output_dim": int(fde_dim),
        "fde_config_hash": fde_config_fingerprint(int(fde_dim)),
    }


def build_prime_disk_index_v2(
    data_root: Path,
    index_dir: Path,
    encoder: QueryEncoderProtocol,
    *,
    corpus_jsonl: Path | None = None,
    batch_size: int = 64,
    max_docs: int = -1,
    rebuild: bool = False,
    build_faiss: bool = False,
) -> None:
    """Build disk index using a QueryEncoderProtocol (JinaVera default)."""
    shim = _HfEncoderShim(encoder)
    build_prime_disk_index(
        data_root,
        index_dir,
        shim,  # type: ignore[arg-type]
        encoder_name=encoder.name,
        corpus_jsonl=corpus_jsonl,
        batch_size=batch_size,
        max_docs=max_docs,
        rebuild=rebuild,
        build_faiss=build_faiss,
        index_metadata=_index_metadata_for_encoder(encoder),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Build STaRK-Prime disk index v2")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/stark_prime"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/stark_prime/index/prime"),
    )
    parser.add_argument(
        "--encoder",
        default="jinavera",
        help="jinavera (default) or contriever",
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=1024,
        help="Jina MRL / FDE target dim (128/256/512/1024/2048)",
    )
    parser.add_argument(
        "--fde-output-dim",
        type=int,
        default=1024,
        help="Jinavera FDE final_projection_dimension",
    )
    parser.add_argument(
        "--legacy-fde-prefix",
        action="store_true",
        help="Full 10240-d FDE + prefix truncate (deprecated)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-docs", type=int, default=-1)
    parser.add_argument(
        "--build-faiss",
        action="store_true",
        help="Build FAISS IndexFlatIP dense indexes",
    )
    args = parser.parse_args()

    fde_output_dim: int | None = args.fde_output_dim
    if args.legacy_fde_prefix:
        fde_output_dim = None

    encoder = create_query_encoder(
        args.encoder,
        truncate_dim=args.truncate_dim,
        fde_output_dim=fde_output_dim,
    )
    rebuild = os.getenv("ASMR_INDEX_REBUILD", "0") == "1"
    build_prime_disk_index_v2(
        args.data_root,
        args.index_dir,
        encoder,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        rebuild=rebuild,
        build_faiss=args.build_faiss,
    )


if __name__ == "__main__":
    main()
