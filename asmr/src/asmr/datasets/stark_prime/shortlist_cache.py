"""Disk cache for STaRK-Prime train/eval shortlists (v1 monolithic + v2 streaming)."""

import argparse
import gc
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from asmr.datasets.stark_prime.loader import PrimeQuery, load_prime_queries
from asmr.datasets.stark_prime.query_cache import QueryEmbeddingCache, resolve_query_emb
from asmr.datasets.stark_prime.torch_dataset import (
    StarkRankingExample,
    _example_from_shortlist,
)
from asmr.evaluation.query_encoders import QueryEncoderProtocol
from asmr.evaluation.stark_prime_disk_index_v2 import PrimeDiskIndexStore

logger = logging.getLogger(__name__)

FORMAT_VERSION_V2 = 2
_LOG_EVERY = 100
_DEFAULT_CHUNK_SIZE = 128


class ShortlistCacheMissingError(FileNotFoundError):
    """Raised when a required on-disk shortlist cache is absent or incomplete."""


@dataclass(frozen=True)
class ShortlistManifest:
    """Metadata for a v2 streaming shortlist cache directory."""

    cache_dir: Path
    split: str
    shortlist_k: int
    encoder: str
    num_examples: int
    chunk_size: int
    num_chunks: int
    complete: bool
    chunks: list[dict[str, Any]] = field(default_factory=list)


def _encoder_slug(encoder_name: str) -> str:
    return encoder_name.replace("/", "_").replace(" ", "_")


def shortlist_cache_path(
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> Path:
    """Return canonical v1 monolithic shortlist cache file path."""
    safe = _encoder_slug(encoder_name)
    return cache_dir / f"shortlist_{split}_k{shortlist_k}_{safe}.pt"


def shortlist_cache_dir_v2(
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> Path:
    """Return v2 streaming cache directory path."""
    safe = _encoder_slug(encoder_name)
    return cache_dir / "shortlist_v2" / f"{split}_k{shortlist_k}_{safe}"


def _manifest_path(cache_dir_v2: Path) -> Path:
    return cache_dir_v2 / "manifest.json"


def _chunks_dir(cache_dir_v2: Path) -> Path:
    return cache_dir_v2 / "chunks"


def _example_to_chunk_row(
    ex: StarkRankingExample,
    query_id: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "query_id": query_id,
        "query_text": ex.query_text,
        "doc_ids": ex.doc_ids,
        "scores": torch.from_numpy(np.asarray(ex.scores, dtype=np.float32)),
        "relevance": torch.from_numpy(np.asarray(ex.relevance, dtype=np.float32)),
    }
    if ex.field_mask is not None:
        row["field_mask"] = torch.from_numpy(np.asarray(ex.field_mask, dtype=bool))
    return row


def _chunk_row_to_example(row: dict[str, object]) -> StarkRankingExample:
    field_mask = None
    if "field_mask" in row and row["field_mask"] is not None:
        mask = row["field_mask"]
        if isinstance(mask, torch.Tensor):
            field_mask = mask.numpy().astype(bool)
        else:
            field_mask = np.array(mask, dtype=bool)
    scores = row["scores"]
    if isinstance(scores, torch.Tensor):
        scores_np = scores.numpy().astype(np.float32)
    else:
        scores_np = np.array(scores, dtype=np.float32)
    relevance = row["relevance"]
    if isinstance(relevance, torch.Tensor):
        rel_np = relevance.numpy().astype(np.float32)
    else:
        rel_np = np.array(relevance, dtype=np.float32)
    return StarkRankingExample(
        query_text=str(row["query_text"]),
        doc_ids=[str(d) for d in row["doc_ids"]],  # type: ignore[index]
        scores=scores_np,
        relevance=rel_np,
        field_mask=field_mask,
    )


def _example_to_dict(ex: StarkRankingExample, query_id: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "query_id": query_id,
        "query_text": ex.query_text,
        "doc_ids": ex.doc_ids,
        "scores": ex.scores.tolist(),
        "relevance": ex.relevance.tolist(),
    }
    if ex.field_mask is not None:
        payload["field_mask"] = ex.field_mask.tolist()
    return payload


def _example_from_dict(row: dict[str, object]) -> StarkRankingExample:
    field_mask = None
    if "field_mask" in row:
        field_mask = np.array(row["field_mask"], dtype=bool)
    return StarkRankingExample(
        query_text=str(row["query_text"]),
        doc_ids=[str(d) for d in row["doc_ids"]],  # type: ignore[index]
        scores=np.array(row["scores"], dtype=np.float32),
        relevance=np.array(row["relevance"], dtype=np.float32),
        field_mask=field_mask,
    )


def _load_chunk_file(path: Path) -> list[StarkRankingExample]:
    data = torch.load(path, weights_only=False)
    rows = data.get("examples", [])
    return [_chunk_row_to_example(row) for row in rows]


def load_shortlist_manifest(
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> ShortlistManifest | None:
    """Load v2 manifest when present; incomplete caches return ``complete=False``."""
    cache_root = shortlist_cache_dir_v2(cache_dir, split, shortlist_k, encoder_name)
    path = _manifest_path(cache_root)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if int(data.get("format_version", 0)) != FORMAT_VERSION_V2:
        return None
    if int(data.get("shortlist_k", -1)) != shortlist_k:
        return None
    if str(data.get("encoder", "")) != encoder_name:
        return None
    return ShortlistManifest(
        cache_dir=cache_root,
        split=str(data.get("split", split)),
        shortlist_k=shortlist_k,
        encoder=encoder_name,
        num_examples=int(data.get("num_examples", 0)),
        chunk_size=int(data.get("chunk_size", _DEFAULT_CHUNK_SIZE)),
        num_chunks=int(data.get("num_chunks", 0)),
        complete=bool(data.get("complete", False)),
        chunks=list(data.get("chunks", [])),
    )


class StreamingStarkRankingDataset(Dataset[StarkRankingExample]):
    """Lazy dataset over v2 shortlist chunk files (bounded in-memory chunk cache)."""

    def __init__(self, manifest: ShortlistManifest) -> None:
        if not manifest.complete:
            msg = f"Shortlist cache incomplete: {manifest.cache_dir}"
            raise ShortlistCacheMissingError(msg)
        self._manifest = manifest
        self._chunk_starts: list[int] = []
        offset = 0
        for chunk in manifest.chunks:
            self._chunk_starts.append(offset)
            offset += int(chunk["count"])
        self._loaded_chunk_index: int | None = None
        self._loaded_examples: list[StarkRankingExample] = []

    def __len__(self) -> int:
        return self._manifest.num_examples

    def _load_chunk(self, chunk_index: int) -> None:
        if self._loaded_chunk_index == chunk_index:
            return
        chunk_meta = self._manifest.chunks[chunk_index]
        chunk_path = self._manifest.cache_dir / str(chunk_meta["file"])
        self._loaded_examples = _load_chunk_file(chunk_path)
        self._loaded_chunk_index = chunk_index

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        for chunk_index in range(len(self._manifest.chunks) - 1, -1, -1):
            if idx >= self._chunk_starts[chunk_index]:
                local = idx - self._chunk_starts[chunk_index]
                return chunk_index, local
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> StarkRankingExample:
        chunk_index, local = self._locate(idx)
        self._load_chunk(chunk_index)
        return self._loaded_examples[local]


class StreamingShortlistWriter:
    """Append shortlist examples and flush fixed-size chunks to disk."""

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        shortlist_k: int,
        encoder_name: str,
        *,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._cache_dir = shortlist_cache_dir_v2(
            cache_dir,
            split,
            shortlist_k,
            encoder_name,
        )
        self._split = split
        self._shortlist_k = shortlist_k
        self._encoder_name = encoder_name
        self._chunk_size = chunk_size
        self._buffer: list[dict[str, object]] = []
        self._chunk_index = 0
        self._num_examples = 0
        self._chunk_meta: list[dict[str, Any]] = []
        self._chunks_path = _chunks_dir(self._cache_dir)
        self._chunks_path.mkdir(parents=True, exist_ok=True)
        self._write_manifest(complete=False)

    def append(self, ex: StarkRankingExample, query_id: int) -> None:
        self._buffer.append(_example_to_chunk_row(ex, query_id))
        self._num_examples += 1
        if len(self._buffer) >= self._chunk_size:
            self.flush_chunk()

    def flush_chunk(self) -> None:
        if not self._buffer:
            return
        chunk_name = f"chunks/chunk_{self._chunk_index:05d}.pt"
        chunk_path = self._cache_dir / chunk_name
        torch.save(
            {
                "chunk_index": self._chunk_index,
                "examples": self._buffer,
            },
            chunk_path,
        )
        qids = [int(row["query_id"]) for row in self._buffer]
        self._chunk_meta.append(
            {
                "file": chunk_name,
                "count": len(self._buffer),
                "query_id_min": min(qids),
                "query_id_max": max(qids),
            },
        )
        logger.info(
            "Wrote shortlist chunk %s (%d examples, total=%d)",
            chunk_name,
            len(self._buffer),
            self._num_examples,
        )
        self._buffer.clear()
        self._chunk_index += 1
        gc.collect()

    def finalize(self) -> Path:
        self.flush_chunk()
        self._write_manifest(complete=True)
        logger.info(
            "Wrote streaming shortlist cache %s (%d examples, %d chunks)",
            self._cache_dir,
            self._num_examples,
            len(self._chunk_meta),
        )
        return self._cache_dir

    def _write_manifest(self, *, complete: bool) -> None:
        payload = {
            "format_version": FORMAT_VERSION_V2,
            "split": self._split,
            "shortlist_k": self._shortlist_k,
            "encoder": self._encoder_name,
            "num_examples": self._num_examples,
            "chunk_size": self._chunk_size,
            "num_chunks": len(self._chunk_meta),
            "complete": complete,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunks": self._chunk_meta,
        }
        path = _manifest_path(self._cache_dir)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(path)


def save_shortlist_cache(
    examples: list[StarkRankingExample],
    queries: list[PrimeQuery],
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> Path:
    """Persist pre-built shortlist examples (v1 monolithic format)."""
    id_by_text = {q.query: q.query_id for q in queries}
    rows: list[dict[str, object]] = []
    for ex in examples:
        qid = id_by_text.get(ex.query_text, -1)
        rows.append(_example_to_dict(ex, qid))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = shortlist_cache_path(cache_dir, split, shortlist_k, encoder_name)
    torch.save(
        {
            "split": split,
            "shortlist_k": shortlist_k,
            "encoder": encoder_name,
            "examples": rows,
        },
        path,
    )
    logger.info("Wrote shortlist cache %s (%d examples)", path, len(rows))
    return path


def _load_v1_shortlist_cache(
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> list[StarkRankingExample] | None:
    path = shortlist_cache_path(cache_dir, split, shortlist_k, encoder_name)
    if not path.exists():
        return None
    data = torch.load(path, weights_only=False)
    if int(data.get("shortlist_k", -1)) != shortlist_k:
        return None
    if str(data.get("encoder", "")) != encoder_name:
        return None
    rows = data.get("examples", [])
    return [_example_from_dict(row) for row in rows]


def _load_v2_shortlist_examples(
    manifest: ShortlistManifest,
) -> list[StarkRankingExample]:
    if not manifest.complete:
        return []
    examples: list[StarkRankingExample] = []
    for chunk in manifest.chunks:
        chunk_path = manifest.cache_dir / str(chunk["file"])
        examples.extend(_load_chunk_file(chunk_path))
    return examples


def load_shortlist_cache(
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> list[StarkRankingExample] | None:
    """Load cached shortlist examples (v2 complete manifest preferred, then v1)."""
    manifest = load_shortlist_manifest(cache_dir, split, shortlist_k, encoder_name)
    if manifest is not None and manifest.complete:
        examples = _load_v2_shortlist_examples(manifest)
        if examples:
            logger.info(
                "Loaded %d %s shortlists from v2 cache (k=%d, encoder=%s)",
                len(examples),
                split,
                shortlist_k,
                encoder_name,
            )
            return examples
    return _load_v1_shortlist_cache(cache_dir, split, shortlist_k, encoder_name)


def is_shortlist_cache_ready(
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> bool:
    """Return whether a complete v2 or v1 shortlist cache exists."""
    manifest = load_shortlist_manifest(cache_dir, split, shortlist_k, encoder_name)
    if manifest is not None and manifest.complete and manifest.num_examples > 0:
        return True
    path = shortlist_cache_path(cache_dir, split, shortlist_k, encoder_name)
    return path.exists()


def build_shortlist_cache_streaming(
    queries: list[PrimeQuery],
    store: PrimeDiskIndexStore,
    encoder: QueryEncoderProtocol,
    caches: dict[str, QueryEmbeddingCache],
    split: str,
    cache_dir: Path,
    *,
    shortlist_k: int,
    parallel: bool = False,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Path:
    """Materialize hybrid shortlists and persist via chunked streaming writer."""
    writer = StreamingShortlistWriter(
        cache_dir,
        split,
        shortlist_k,
        encoder.name,
        chunk_size=chunk_size,
    )
    total = len(queries)
    for qi, query in enumerate(queries):
        q_emb = resolve_query_emb(query, encoder, caches, split)
        doc_ids, scores, mask = store.shortlist_hybrid_dispatch(
            query.query,
            q_emb,
            shortlist_k,
            parallel=parallel,
        )
        if not doc_ids:
            continue
        ex = _example_from_shortlist(query, doc_ids, scores, mask)
        writer.append(ex, query.query_id)
        if (qi + 1) % _LOG_EVERY == 0 or qi + 1 == total:
            logger.info("%s shortlist %d/%d", split, qi + 1, total)
    return writer.finalize()


def build_shortlist_cache(
    queries: list[PrimeQuery],
    store: PrimeDiskIndexStore,
    encoder: QueryEncoderProtocol,
    caches: dict[str, QueryEmbeddingCache],
    split: str,
    cache_dir: Path,
    *,
    shortlist_k: int,
    parallel: bool = False,
    streaming: bool = False,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Path:
    """Materialize hybrid shortlists; v2 streaming when ``streaming=True``."""
    if streaming:
        return build_shortlist_cache_streaming(
            queries,
            store,
            encoder,
            caches,
            split,
            cache_dir,
            shortlist_k=shortlist_k,
            parallel=parallel,
            chunk_size=chunk_size,
        )

    examples: list[dict[str, object]] = []
    total = len(queries)
    for qi, query in enumerate(queries):
        q_emb = resolve_query_emb(query, encoder, caches, split)
        doc_ids, scores, mask = store.shortlist_hybrid_dispatch(
            query.query,
            q_emb,
            shortlist_k,
            parallel=parallel,
        )
        if not doc_ids:
            continue
        ex = _example_from_shortlist(query, doc_ids, scores, mask)
        examples.append(_example_to_dict(ex, query.query_id))
        if (qi + 1) % _LOG_EVERY == 0 or qi + 1 == total:
            logger.info("shortlist cache %s %d/%d", split, qi + 1, total)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = shortlist_cache_path(cache_dir, split, shortlist_k, encoder.name)
    torch.save(
        {
            "split": split,
            "shortlist_k": shortlist_k,
            "encoder": encoder.name,
            "examples": examples,
        },
        path,
    )
    logger.info("Wrote shortlist cache %s (%d examples)", path, len(examples))
    return path


def spawn_build_shortlist_cache(
    *,
    data_root: Path,
    index_dir: Path,
    encoder_name: str,
    split: str = "train",
    shortlist_k: int = 100,
    parallel: bool = False,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    hf_model_name: str = "facebook/contriever-msmarco",
    max_queries: int = -1,
) -> None:
    """Run shortlist cache builder in a subprocess (Process A)."""
    cmd = [
        sys.executable,
        "-m",
        "asmr.datasets.stark_prime.shortlist_cache",
        "--data-root",
        str(data_root),
        "--index-dir",
        str(index_dir),
        "--split",
        split,
        "--shortlist-k",
        str(shortlist_k),
        "--encoder",
        encoder_name,
        "--streaming",
        "--chunk-size",
        str(chunk_size),
        "--hf-model-name",
        hf_model_name,
    ]
    if parallel:
        cmd.append("--parallel")
    if max_queries > 0:
        cmd.extend(["--max-queries", str(max_queries)])
    logger.info("Spawning shortlist cache builder: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Build STaRK-Prime shortlist cache")
    parser.add_argument("--data-root", type=Path, default=Path("data/stark_prime"))
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/stark_prime/index/prime"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--encoder", default="contriever")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Write v2 chunked streaming cache (recommended)",
    )
    parser.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-queries", type=int, default=-1)
    parser.add_argument(
        "--hf-model-name",
        default="facebook/contriever-msmarco",
    )
    args = parser.parse_args()

    from asmr.datasets.stark_prime.query_cache import load_query_caches
    from asmr.evaluation.query_encoders import create_query_encoder

    encoder = create_query_encoder(
        args.encoder,
        hf_model_name=args.hf_model_name,
    )
    store = PrimeDiskIndexStore(args.index_dir)
    cache_dir = args.data_root / "cache"
    caches = load_query_caches(cache_dir, encoder_name=encoder.name)
    queries = load_prime_queries(args.data_root, args.split)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    build_shortlist_cache(
        queries,
        store,
        encoder,
        caches,
        args.split,
        cache_dir,
        shortlist_k=args.shortlist_k,
        parallel=args.parallel,
        streaming=args.streaming,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
