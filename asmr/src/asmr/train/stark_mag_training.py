"""MFARAll training helpers for STaRK-MAG (Phase 1 head-only)."""

import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import optim

from asmr.datasets.query_cache import QueryEmbeddingCache, resolve_query_emb
from asmr.datasets.stark_mag.loader import MAG_FIELD_NAMES, MagQuery
from asmr.datasets.stark_mag.shortlist_cache import (
    load_shortlist_cache,
    save_shortlist_cache,
)
from asmr.datasets.stark_prime.torch_dataset import (
    StarkRankingDataset,
    StarkRankingExample,
    collate_stark_batch,
)
from asmr.encode.protocol import NamedTextEncoderProtocol
from asmr.evaluation.stark_disk_index import StarkDiskIndexStore
from asmr.train.aggregation import MFARFieldAdapter
from asmr.train.config import TrainConfig, TrainMfarConfig
from asmr.train.trainer import AggregationTrainer


def _build_mag_example(
    query: MagQuery,
    doc_ids: list[str],
    scores: np.ndarray,
    mask: np.ndarray,
) -> StarkRankingExample:
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


logger = logging.getLogger(__name__)

_LOG_EVERY = 50

GATE_RATIO_90_MAG = {
    "hit@1": 0.90,
    "recall@20": 0.90,
    "mrr": 0.90,
}


def build_mag_training_examples(
    train_queries: list[MagQuery],
    indexes: StarkDiskIndexStore,
    encoder: NamedTextEncoderProtocol,
    *,
    shortlist_k: int,
    cache_dir: Optional[Path] = None,
    rebuild_cache: bool = False,
    encoder_name: str = "unknown",
    query_caches: dict[str, QueryEmbeddingCache] | None = None,
) -> list[StarkRankingExample]:
    """Build training shortlists for all queries, with optional pickle cache.

    Args:
        train_queries: MAG train split queries.
        indexes: Per-field BM25 + FAISS indexes.
        encoder: Text encoder for query embeddings.
        shortlist_k: Candidates per field×scorer.
        cache_dir: If set, load from / save to a pickle cache here.
        rebuild_cache: Force rebuild even if cache file exists.
        encoder_name: Used as part of the cache filename.
        query_caches: Pre-built query embedding caches keyed by split name.

    Returns:
        List of StarkRankingExample (one per query with non-empty shortlist).
    """
    if cache_dir is not None and not rebuild_cache:
        cached = load_shortlist_cache(cache_dir, "train", shortlist_k, encoder_name)
        if cached is not None:
            return cached

    _caches: dict[str, QueryEmbeddingCache] = query_caches or {}
    examples: list[StarkRankingExample] = []
    total = len(train_queries)
    t0 = time.time()
    for qi, query in enumerate(train_queries):
        q_emb = resolve_query_emb(query, encoder, _caches, "train")
        doc_ids, scores, mask = indexes.shortlist_hybrid(
            query.query, q_emb, shortlist_k
        )
        if doc_ids:
            examples.append(_build_mag_example(query, doc_ids, scores, mask))
        if (qi + 1) % _LOG_EVERY == 0 or qi + 1 == total:
            logger.info(
                "MAG shortlist %d/%d elapsed=%.0fs",
                qi + 1,
                total,
                time.time() - t0,
            )

    if cache_dir is not None:
        save_shortlist_cache(examples, cache_dir, "train", shortlist_k, encoder_name)

    return examples


def train_mfar_mag(
    train_queries: list[MagQuery],
    indexes: StarkDiskIndexStore,
    encoder: NamedTextEncoderProtocol,
    cfg: TrainMfarConfig,
    device: torch.device,
    *,
    cache_dir: Optional[Path] = None,
    rebuild_cache: bool = False,
    encoder_name: str = "unknown",
    query_caches: dict[str, QueryEmbeddingCache] | None = None,
) -> MFARFieldAdapter:
    """Train MFARFieldAdapter for STaRK-MAG (Phase 1: head-only).

    Args:
        train_queries: MAG train split queries.
        indexes: Per-field BM25 + FAISS indexes (built on full corpus).
        encoder: Frozen text encoder used for query embedding.
        cfg: Training hyperparameters (shortlist_k, epochs, adapter_lr, …).
        device: torch device.
        cache_dir: Optional path for shortlist pickle cache.
        rebuild_cache: Force shortlist rebuild even if cached.
        encoder_name: Encoder identifier for cache filename.

    Returns:
        Trained MFARFieldAdapter ready for inference.
    """
    f_num = len(MAG_FIELD_NAMES)
    adapter = MFARFieldAdapter(encoder.embedding_dim, f_num, 2)
    adapter.to(device)
    train_cfg = TrainConfig(query_dim=encoder.embedding_dim)
    trainer = AggregationTrainer(adapter, train_cfg)
    opt = optim.AdamW(
        adapter.parameters(),
        lr=cfg.adapter_lr,
        weight_decay=cfg.weight_decay,
    )

    examples = build_mag_training_examples(
        train_queries,
        indexes,
        encoder,
        shortlist_k=cfg.shortlist_k,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
        encoder_name=encoder_name,
        query_caches=query_caches,
    )

    ds = StarkRankingDataset(examples)
    indices = list(range(len(ds)))
    logger.info(
        "Training MFARFieldAdapter (MAG) on %d shortlists, epochs=%d",
        len(ds),
        cfg.epochs,
    )

    for epoch in range(1, cfg.epochs + 1):
        random.shuffle(indices)
        loss_sum = 0.0
        for idx in indices:
            batch = collate_stark_batch([ds[idx]], encoder, device)
            opt.zero_grad()
            out = trainer.training_step(batch)
            out.loss.backward()
            opt.step()
            loss_sum += out.loss.item()
        logger.info(
            "Epoch %d/%d loss=%.4f",
            epoch,
            cfg.epochs,
            loss_sum / max(len(indices), 1),
        )
    return adapter


def check_gate_90_mag(ratio: dict[str, float]) -> tuple[bool, dict[str, bool]]:
    """Return whether each MAG metric meets the 90% paper ratio gate."""
    checks = {
        key: ratio.get(key, 0.0) >= threshold
        for key, threshold in GATE_RATIO_90_MAG.items()
    }
    return all(checks.values()), checks
