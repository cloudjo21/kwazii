"""Shared query embedding cache for STaRK datasets (MAG, Prime, …).

On-disk layout (memmap backend):
    {cache_dir}/query_emb_{split}_{safe}.ids.npy   # int64 [N]
    {cache_dir}/query_emb_{split}_{safe}.emb.npy   # float32 [N, D]

Backward-compat: falls back to reading legacy .pt files if .npy pair is absent.
"""

import logging
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import numpy.typing as npt

from asmr.encode.protocol import NamedTextEncoderProtocol
from fde.config import PromptType

logger = logging.getLogger(__name__)


@runtime_checkable
class QueryProtocol(Protocol):
    """Structural protocol satisfied by MagQuery, PrimeQuery, etc."""

    @property
    def query_id(self) -> int: ...

    @property
    def query(self) -> str: ...


def _encoder_slug(encoder_name: str) -> str:
    return encoder_name.replace("/", "_").replace(" ", "_")


def _ids_path(cache_dir: Path, split: str, encoder_name: str) -> Path:
    safe = _encoder_slug(encoder_name)
    return cache_dir / f"query_emb_{split}_{safe}.ids.npy"


def _emb_path(cache_dir: Path, split: str, encoder_name: str) -> Path:
    safe = _encoder_slug(encoder_name)
    return cache_dir / f"query_emb_{split}_{safe}.emb.npy"


def is_query_cache_built(cache_dir: Path, split: str, encoder_name: str) -> bool:
    """Return True iff both .npy files for this (split, encoder) exist."""
    return (
        _ids_path(cache_dir, split, encoder_name).exists()
        and _emb_path(cache_dir, split, encoder_name).exists()
    )


def build_query_emb_cache(
    queries: Sequence[QueryProtocol],
    encoder: NamedTextEncoderProtocol,
    cache_dir: Path,
    split: str,
    *,
    batch_size: int = 64,
    force: bool = False,
) -> tuple[Path, Path]:
    """Encode all queries and persist as memmap-friendly .npy files.

    Skips if files already exist unless force=True. Writes atomically via
    .tmp files to avoid half-written caches.
    """
    ids_p = _ids_path(cache_dir, split, encoder.name)
    emb_p = _emb_path(cache_dir, split, encoder.name)
    if not force and ids_p.exists() and emb_p.exists():
        logger.info("Query cache hit — skipping build: %s", emb_p)
        return ids_p, emb_p

    if not queries:
        logger.warning(
            "build_query_emb_cache called with empty query list for split=%s", split
        )
        return ids_p, emb_p

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Building query cache: split=%s encoder=%s n=%d",
        split,
        encoder.name,
        len(queries),
    )

    ids: list[int] = [q.query_id for q in queries]
    texts: list[str] = [q.query for q in queries]

    chunks: list[npt.NDArray[np.float32]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        chunks.append(
            encoder.encode_text(batch, PromptType.QUERY, batch_size=batch_size)
        )
    stacked: npt.NDArray[np.float32] = np.concatenate(chunks, axis=0)

    ids_tmp = ids_p.with_suffix(".npy.tmp")
    emb_tmp = emb_p.with_suffix(".npy.tmp")
    np.save(str(ids_tmp), np.array(ids, dtype=np.int64))
    np.save(str(emb_tmp), stacked)
    ids_tmp.replace(ids_p)
    emb_tmp.replace(emb_p)

    logger.info("Wrote query cache: %s  %s", ids_p, emb_p)
    return ids_p, emb_p


class QueryEmbeddingCache:
    """O(1) memmap-backed lookup of precomputed query embeddings by query_id."""

    def __init__(self, cache_dir: Path, split: str, encoder_name: str) -> None:
        ids_p = _ids_path(cache_dir, split, encoder_name)
        emb_p = _emb_path(cache_dir, split, encoder_name)
        ids: npt.NDArray[np.int64] = np.load(str(ids_p))
        # mmap_mode='r': OS pages only accessed rows — full array never in RAM
        self._emb: npt.NDArray[np.float32] = np.load(str(emb_p), mmap_mode="r")
        self._id_to_idx: dict[int, int] = {int(qid): i for i, qid in enumerate(ids)}

    @classmethod
    def _from_dict(
        cls, id_to_emb: dict[int, npt.NDArray[np.float32]]
    ) -> "QueryEmbeddingCache":
        """Build an in-memory cache from a preloaded id→embedding dict (legacy .pt path)."""
        obj = object.__new__(cls)
        obj._id_to_emb_dict = id_to_emb  # type: ignore[attr-defined]
        obj._emb = None
        obj._id_to_idx = {}
        return obj

    def get(self, query_id: int) -> npt.NDArray[np.float32] | None:
        # Legacy in-memory path (from _from_dict)
        legacy: dict[int, npt.NDArray[np.float32]] | None = getattr(
            self, "_id_to_emb_dict", None
        )
        if legacy is not None:
            return legacy.get(query_id)
        # Memmap path
        idx = self._id_to_idx.get(query_id)
        if idx is None:
            return None
        return np.array(self._emb[idx], dtype=np.float32)

    def get_or_encode(
        self,
        query: QueryProtocol,
        encoder: NamedTextEncoderProtocol,
    ) -> npt.NDArray[np.float32]:
        cached = self.get(query.query_id)
        if cached is not None:
            return cached
        return encoder.encode_text([query.query], PromptType.QUERY)[0]


def load_query_caches(
    cache_dir: Path,
    splits: tuple[str, ...] = ("train", "test"),
    *,
    encoder_name: str,
) -> dict[str, QueryEmbeddingCache]:
    """Load split caches that exist on disk. Silently skips missing splits.

    Tries new .npy memmap format first; falls back to legacy .pt torch checkpoint.
    """
    caches: dict[str, QueryEmbeddingCache] = {}
    safe = _encoder_slug(encoder_name)
    for split in splits:
        ids_p = _ids_path(cache_dir, split, encoder_name)
        emb_p = _emb_path(cache_dir, split, encoder_name)
        if ids_p.exists() and emb_p.exists():
            caches[split] = QueryEmbeddingCache(cache_dir, split, encoder_name)
            logger.info("Loaded query cache (memmap) split=%s  %s", split, emb_p)
            continue
        # Backward-compat: legacy .pt files
        for pt_name in (
            f"query_emb_{split}_{safe}.pt",
            f"query_emb_{split}.pt",
        ):
            pt_path = cache_dir / pt_name
            if pt_path.exists():
                try:
                    import torch

                    data = torch.load(pt_path, weights_only=True)
                    pt_ids: list[int] = list(data["ids"])
                    emb_arr = data["emb"]
                    if hasattr(emb_arr, "numpy"):
                        emb_arr = emb_arr.numpy()
                    emb_arr = np.asarray(emb_arr, dtype=np.float32)
                    id_to_emb = {qid: emb_arr[i] for i, qid in enumerate(pt_ids)}
                    caches[split] = QueryEmbeddingCache._from_dict(id_to_emb)
                    logger.info(
                        "Loaded query cache (legacy .pt) split=%s  %s", split, pt_path
                    )
                except Exception as exc:
                    logger.warning("Failed to load legacy cache %s: %s", pt_path, exc)
                break
    return caches


def resolve_query_emb(
    query: QueryProtocol,
    encoder: NamedTextEncoderProtocol,
    caches: dict[str, QueryEmbeddingCache],
    split: str,
) -> npt.NDArray[np.float32]:
    """Return cached embedding or encode on-the-fly."""
    cache = caches.get(split)
    if cache is not None:
        return cache.get_or_encode(query, encoder)
    return encoder.encode_text([query.query], PromptType.QUERY)[0]
