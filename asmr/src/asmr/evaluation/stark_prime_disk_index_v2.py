"""STaRK-Prime disk index v2 — asmr module integration + QueryRouter shortlist."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np
import torch

from asmr.evaluation.query_encoders import QueryEncoderProtocol, create_query_encoder
from fde.config import fde_config_fingerprint
from asmr.evaluation.stark_prime_disk_index import (
    PrimeDiskIndexStore as _PrimeDiskIndexStoreV1,
    ShortlistTiming,
    build_prime_disk_index,
    migrate_index_v21,
)
from asmr.evaluation.stark_prime_router import shortlist_prime_hybrid_router

logger = logging.getLogger(__name__)

_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    """Run one coroutine on a reused loop (avoids asyncio.run per-query leaks)."""
    global _ASYNC_LOOP
    if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
        _ASYNC_LOOP = asyncio.new_event_loop()
    return _ASYNC_LOOP.run_until_complete(coro)


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


class PrimeDiskIndexStore(_PrimeDiskIndexStoreV1):
    """Disk-backed store with QueryRouter hybrid shortlist (v3 default)."""

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
        "--migrate-v21",
        action="store_true",
        help="Add FAISS dense + doc_id_mapping.marisa to existing v2 index",
    )
    parser.add_argument(
        "--build-faiss",
        action="store_true",
        help="Build FAISS IndexFlatIP dense indexes",
    )
    args = parser.parse_args()

    if args.migrate_v21:
        migrate_index_v21(args.index_dir)
        return

    fde_output_dim: int | None = args.fde_output_dim
    if args.legacy_fde_prefix:
        fde_output_dim = None

    encoder = create_query_encoder(
        args.encoder,
        truncate_dim=args.truncate_dim,
        fde_output_dim=fde_output_dim,
    )
    rebuild = False
    import os

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
