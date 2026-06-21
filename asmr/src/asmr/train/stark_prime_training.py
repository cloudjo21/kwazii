"""MFARAll training helpers for STaRK-Prime v3 (Phase 1 + Phase 2)."""

import logging
import random
import time
from pathlib import Path

from asmr.train.config import TrainMfarConfig, TrainingPhase

import torch
from torch import optim
from torch.utils.data import Dataset

from asmr.datasets.stark_prime.loader import PRIME_FIELD_NAMES, PrimeQuery
from asmr.datasets.stark_prime.query_cache import QueryEmbeddingCache, resolve_query_emb
from asmr.datasets.stark_prime.shortlist_cache import (
    ShortlistCacheMissingError,
    StreamingShortlistWriter,
    StreamingStarkRankingDataset,
    is_shortlist_cache_ready,
    load_shortlist_cache,
    load_shortlist_manifest,
    save_shortlist_cache,
)
from asmr.datasets.stark_prime.torch_dataset import (
    StarkRankingDataset,
    StarkRankingExample,
    _example_from_shortlist,
    collate_stark_batch,
)
from asmr.evaluation.query_encoders import QueryEncoderProtocol, _JinaLoraEncoderWrapper
from asmr.evaluation.stark_prime_disk_index_v2 import PrimeDiskIndexStore
from asmr.train.aggregation import MFARFieldAdapter
from asmr.train.config import TrainConfig
from asmr.train.query_encoder import JinaLoraQueryEncoder
from asmr.train.trainer import AggregationTrainer

logger = logging.getLogger(__name__)

_LOG_EVERY = 100

GATE_RATIO_90 = {
    "hit@1": 0.90,
    "recall@20": 0.90,
    "mrr": 0.90,
}


def resolve_training_phase(
    requested: TrainingPhase,
    encoder: QueryEncoderProtocol,
) -> TrainingPhase:
    """Downgrade to Phase 1 when encoder cannot be fine-tuned."""
    if requested == TrainingPhase.JOINT and not encoder.supports_joint_training():
        logger.warning(
            "Encoder %s does not support Phase 2 joint FT; using Phase 1",
            encoder.name,
        )
        return TrainingPhase.HEAD_ONLY
    return requested


def resolve_train_dataset(
    train_queries: list[PrimeQuery],
    store: PrimeDiskIndexStore | None,
    encoder: QueryEncoderProtocol,
    caches: dict[str, QueryEmbeddingCache],
    *,
    shortlist_k: int,
    parallel: bool,
    cache_dir: Path | None = None,
    rebuild_shortlist_cache: bool = False,
    require_shortlist_cache: bool = False,
    use_streaming_cache: bool = True,
    chunk_size: int = 128,
    log_timing: bool = False,
) -> Dataset[StarkRankingExample]:
    """Resolve training dataset from disk cache or inline shortlist build."""
    if cache_dir is not None and not rebuild_shortlist_cache:
        if require_shortlist_cache and not is_shortlist_cache_ready(
            cache_dir,
            "train",
            shortlist_k,
            encoder.name,
        ):
            msg = (
                f"Required shortlist cache missing for encoder={encoder.name}, "
                f"k={shortlist_k} under {cache_dir}"
            )
            raise ShortlistCacheMissingError(msg)

        manifest = load_shortlist_manifest(
            cache_dir,
            "train",
            shortlist_k,
            encoder.name,
        )
        if manifest is not None and manifest.complete and use_streaming_cache:
            logger.info(
                "Using streaming shortlist dataset (%d examples, %d chunks)",
                manifest.num_examples,
                manifest.num_chunks,
            )
            return StreamingStarkRankingDataset(manifest)

        cached = load_shortlist_cache(
            cache_dir,
            "train",
            shortlist_k,
            encoder.name,
        )
        if cached is not None:
            logger.info(
                "Loaded %d train shortlists from cache (k=%d, encoder=%s)",
                len(cached),
                shortlist_k,
                encoder.name,
            )
            return StarkRankingDataset(cached)

        if require_shortlist_cache:
            msg = (
                f"Required shortlist cache missing for encoder={encoder.name}, "
                f"k={shortlist_k} under {cache_dir}"
            )
            raise ShortlistCacheMissingError(msg)

    if store is None:
        msg = "PrimeDiskIndexStore required to build train shortlists inline"
        raise ValueError(msg)

    examples: list[StarkRankingExample] = []
    total = len(train_queries)
    t0 = time.time()
    for qi, query in enumerate(train_queries):
        q_emb = resolve_query_emb(query, encoder, caches, "train")
        doc_ids, scores, mask = store.shortlist_hybrid_dispatch(
            query.query,
            q_emb,
            shortlist_k,
            parallel=parallel,
            log_timing=log_timing and qi == 0,
        )
        if doc_ids:
            examples.append(
                _example_from_shortlist(query, doc_ids, scores, mask),
            )
        if (qi + 1) % _LOG_EVERY == 0 or qi + 1 == total:
            logger.info(
                "train shortlist %d/%d, elapsed=%.0fs",
                qi + 1,
                total,
                time.time() - t0,
            )

    if cache_dir is not None:
        if use_streaming_cache:
            id_by_text = {q.query: q.query_id for q in train_queries}
            writer = StreamingShortlistWriter(
                cache_dir,
                "train",
                shortlist_k,
                encoder.name,
                chunk_size=chunk_size,
            )
            for ex in examples:
                qid = id_by_text.get(ex.query_text, -1)
                writer.append(ex, qid)
            writer.finalize()
            manifest = load_shortlist_manifest(
                cache_dir,
                "train",
                shortlist_k,
                encoder.name,
            )
            if manifest is not None and manifest.complete:
                return StreamingStarkRankingDataset(manifest)
            msg = "Streaming shortlist cache finalize failed after inline build"
            raise RuntimeError(msg)

        save_shortlist_cache(
            examples,
            train_queries,
            cache_dir,
            "train",
            shortlist_k,
            encoder.name,
        )
    return StarkRankingDataset(examples)


def _build_optimizer(
    adapter: MFARFieldAdapter,
    encoder: QueryEncoderProtocol,
    cfg: TrainMfarConfig,
    phase: TrainingPhase,
    device: torch.device,
) -> optim.AdamW:
    if phase == TrainingPhase.JOINT:
        inner: JinaLoraQueryEncoder | None = None
        if isinstance(encoder, JinaLoraQueryEncoder):
            inner = encoder
        elif isinstance(encoder, _JinaLoraEncoderWrapper):
            inner = encoder.inner
        elif hasattr(encoder, "inner") and isinstance(
            encoder.inner,
            JinaLoraQueryEncoder,
        ):
            inner = encoder.inner
        if inner is not None:
            inner.trainable_module().to(device)
            lora_params = inner.lora_parameters()
            return optim.AdamW(
                [
                    {"params": adapter.parameters(), "lr": cfg.adapter_lr},
                    {"params": lora_params, "lr": cfg.encoder_lr},
                ],
                weight_decay=cfg.weight_decay,
            )
        enc_module = encoder.trainable_module()
        enc_module.to(device)
        return optim.AdamW(
            [
                {"params": adapter.parameters(), "lr": cfg.adapter_lr},
                {"params": enc_module.parameters(), "lr": cfg.encoder_lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    return optim.AdamW(
        adapter.parameters(),
        lr=cfg.adapter_lr,
        weight_decay=cfg.weight_decay,
    )


def train_mfar(
    train_queries: list[PrimeQuery],
    store: PrimeDiskIndexStore | None,
    encoder: QueryEncoderProtocol,
    caches: dict[str, QueryEmbeddingCache],
    cfg: TrainMfarConfig,
    device: torch.device,
    *,
    parallel: bool = False,
    cache_dir: Path | None = None,
    rebuild_shortlist_cache: bool = False,
    require_shortlist_cache: bool = False,
    use_streaming_cache: bool = True,
    chunk_size: int = 128,
    log_timing: bool = False,
) -> MFARFieldAdapter:
    """Train MFARFieldAdapter (Phase 1 or Phase 2 joint)."""
    phase = resolve_training_phase(cfg.phase, encoder)
    f_num = len(PRIME_FIELD_NAMES)
    adapter = MFARFieldAdapter(encoder.embedding_dim, f_num, 2).to(device)
    train_cfg = TrainConfig(
        query_dim=encoder.embedding_dim,
        normalize_scores=cfg.normalize_scores,
    )
    trainer = AggregationTrainer(adapter, train_cfg)
    opt = _build_optimizer(adapter, encoder, cfg, phase, device)

    ds = resolve_train_dataset(
        train_queries,
        store,
        encoder,
        caches,
        shortlist_k=cfg.shortlist_k,
        parallel=parallel,
        cache_dir=cache_dir,
        rebuild_shortlist_cache=rebuild_shortlist_cache,
        require_shortlist_cache=require_shortlist_cache,
        use_streaming_cache=use_streaming_cache,
        chunk_size=chunk_size,
        log_timing=log_timing,
    )
    indices = list(range(len(ds)))
    logger.info(
        "Training MFAR (%s) on %d shortlists, epochs=%d",
        phase.name,
        len(ds),
        cfg.epochs,
    )

    train_encoder = phase == TrainingPhase.JOINT
    for epoch in range(1, cfg.epochs + 1):
        random.shuffle(indices)
        loss_sum = 0.0
        for idx in indices:
            batch = collate_stark_batch(
                [ds[idx]],
                encoder,
                device,
                train_encoder=train_encoder,
            )
            opt.zero_grad()
            out = trainer.training_step(batch)
            out.loss.backward()
            opt.step()
            loss_sum += out.loss.item()
        logger.info(
            "Epoch %d/%d loss=%.4f phase=%s",
            epoch,
            cfg.epochs,
            loss_sum / max(len(indices), 1),
            phase.name,
        )
    return adapter


def check_gate_90(ratio: dict[str, float]) -> tuple[bool, dict[str, bool]]:
    """Return whether each metric meets the 90%% paper ratio gate."""
    checks = {
        key: ratio.get(key, 0.0) >= threshold
        for key, threshold in GATE_RATIO_90.items()
    }
    return all(checks.values()), checks
