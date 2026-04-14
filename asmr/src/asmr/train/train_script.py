"""Training entrypoint for aggregation head (Phase 1, §13.4)."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import optim

from asmr.train.aggregation import AggregationHead, MFARFieldAdapter
from asmr.train.config import TrainConfig
from asmr.train.data_stark import (
    StarkRankingDataset,
    StarkRankingExample,
    collate_stark_batch,
)
from asmr.train.query_encoder import HfQueryEncoder
from asmr.train.trainer import AggregationTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dummy data helpers
# ---------------------------------------------------------------------------


def _build_dummy_example(
    f_num: int,
    m_num: int,
    d_num: int,
    query_idx: int = 0,
) -> StarkRankingExample:
    scores = np.random.randn(f_num, m_num, d_num).astype(np.float32)
    rel = np.zeros(d_num, dtype=np.float32)
    rel[random.randint(0, d_num - 1)] = 1.0
    return StarkRankingExample(
        query_text=f"dummy query {query_idx}",
        doc_ids=[f"d{i}" for i in range(d_num)],
        scores=scores,
        relevance=rel,
    )


class _DummyEncoder:
    """Matches HfQueryEncoder.encode API — no HF download needed."""

    def __init__(self, dim: int, device: torch.device) -> None:
        self._dim = dim
        self._device = device

    def encode(self, texts: list[str]) -> torch.Tensor:
        del texts
        return torch.randn(1, self._dim, device=self._device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train aggregation head (Phase 1).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model",
        choices=["adapter", "head"],
        default="adapter",
        help="adapter=MFARFieldAdapter (linear G), head=AggregationHead (MLP G_θ)",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use random tensors — no HF download, no data file needed.",
    )
    parser.add_argument(
        "--num-dummy",
        type=int,
        default=20,
        metavar="N",
        help="Number of dummy examples to generate (default: 20).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to training JSONL file (StarkRankingDataset format).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="facebook/contriever-msmarco",
        help="HuggingFace model ID for frozen query encoder.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--temperature", type=float, default=0.07, help="Listwise loss temperature."
    )
    parser.add_argument(
        "--normalize-scores",
        action="store_true",
        help="Apply per-(field,scorer) batch normalization before head.",
    )
    args = parser.parse_args()

    if not args.dummy and args.data is None:
        parser.error("Provide --data PATH or use --dummy.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------ data
    f_num, m_num, d_num = 3, 2, 50
    if args.dummy:
        logger.info(
            "Building %d dummy examples (F=%d, M=%d, D=%d)",
            args.num_dummy,
            f_num,
            m_num,
            d_num,
        )
        examples = [
            _build_dummy_example(f_num, m_num, d_num, i) for i in range(args.num_dummy)
        ]
        ds = StarkRankingDataset(examples)
    else:
        ds = StarkRankingDataset.from_jsonl(args.data)
        # infer dims from first example
        ex0 = ds[0]
        s = ex0.scores
        if s.ndim == 3:
            f_num, m_num, d_num = s.shape
        elif s.ndim == 2:
            f_num, d_num = s.shape
            m_num = 1
        else:
            raise ValueError(f"Unexpected scores ndim: {s.ndim}")
        logger.info(
            "Dataset: %d examples, F=%d, M=%d, D=%d", len(ds), f_num, m_num, d_num
        )

    # --------------------------------------------------------------- encoder
    if args.dummy:
        query_dim = 768
        encoder = _DummyEncoder(query_dim, device)
        logger.info("Encoder: DummyEncoder (dim=%d)", query_dim)
    else:
        logger.info("Loading encoder: %s", args.encoder)
        encoder = HfQueryEncoder(model_name=args.encoder, device=str(device))
        query_dim = encoder.embedding_dim
        logger.info("Encoder loaded: dim=%d", query_dim)

    # ----------------------------------------------------------------- model
    cfg = TrainConfig(
        query_dim=query_dim,
        temperature=args.temperature,
        normalize_scores=args.normalize_scores,
    )
    if args.model == "adapter":
        model: torch.nn.Module = MFARFieldAdapter(query_dim, f_num, m_num).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Model: MFARFieldAdapter | params=%d", param_count)
    else:
        model = AggregationHead(
            query_dim=query_dim,
            num_fields=f_num,
            num_scorers=m_num,
            hidden_dim=cfg.hidden_dim,
            aux_dim=cfg.aux_dim,
        ).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "Model: AggregationHead (hidden=%d, aux=%d) | params=%d",
            cfg.hidden_dim,
            cfg.aux_dim,
            param_count,
        )

    trainer = AggregationTrainer(model, cfg)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # -------------------------------------------------------------- training
    logger.info(
        "Training: epochs=%d, lr=%g, temperature=%g, normalize_scores=%s",
        args.epochs,
        args.lr,
        args.temperature,
        args.normalize_scores,
    )
    indices = list(range(len(ds)))
    for epoch in range(1, args.epochs + 1):
        random.shuffle(indices)
        epoch_loss = 0.0
        for step, idx in enumerate(indices):
            batch = collate_stark_batch([ds[idx]], encoder, device)
            opt.zero_grad()
            out = trainer.training_step(batch)
            out.loss.backward()
            opt.step()
            epoch_loss += out.loss.item()

        avg_loss = epoch_loss / max(len(indices), 1)
        logger.info("Epoch %3d/%d | loss=%.6f", epoch, args.epochs, avg_loss)

    logger.info("Training complete.")

    # ----------------------------------------------------------- checkpoint
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg,
                "model_type": args.model,
                "f_num": f_num,
                "m_num": m_num,
                "encoder": args.encoder,
            },
            args.checkpoint,
        )
        logger.info("Checkpoint saved: %s", args.checkpoint)

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        logger.info("CUDA peak memory: %.1f MB", peak)


if __name__ == "__main__":
    main()
