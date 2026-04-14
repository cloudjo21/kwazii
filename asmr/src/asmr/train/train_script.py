"""Minimal training entrypoint for aggregation head (§13.4)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim

from asmr.train.aggregation import MFARFieldAdapter
from asmr.train.config import TrainConfig
from asmr.train.data_stark import (
    StarkRankingDataset,
    StarkRankingExample,
    collate_stark_batch,
)
from asmr.train.query_encoder import HfQueryEncoder
from asmr.train.trainer import AggregationTrainer


def _build_dummy_example(
    f_num: int,
    m_num: int,
    d_num: int,
) -> StarkRankingExample:
    scores = torch.randn(f_num, m_num, d_num).float().numpy()
    rel = torch.zeros(d_num).float().numpy()
    rel[0] = 1.0
    return StarkRankingExample(
        query_text="dummy query",
        doc_ids=[f"d{i}" for i in range(d_num)],
        scores=scores,
        relevance=rel,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MFARFieldAdapter (smoke/dummy).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dummy", action="store_true", help="Use random tensors (no HF download).")
    parser.add_argument("--encoder", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    f_num, m_num, d_num = 2, 2, 5
    if args.dummy:
        hidden = 32
        ex = _build_dummy_example(f_num, m_num, d_num)
        ds = StarkRankingDataset([ex])
        batch = collate_stark_batch([ds[0]], _DummyEncoder(32, device), device)
    else:
        encoder = HfQueryEncoder(model_name=args.encoder, device=str(device))
        ex = _build_dummy_example(f_num, m_num, d_num)
        ds = StarkRankingDataset([ex])
        batch = collate_stark_batch([ds[0]], encoder, device)

    cfg = TrainConfig(query_dim=batch.query_emb.shape[-1], temperature=0.1)
    model = MFARFieldAdapter(cfg.query_dim, f_num, m_num).to(device)
    trainer = AggregationTrainer(model, cfg)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    for _ in range(args.epochs):
        opt.zero_grad()
        out = trainer.training_step(batch)
        out.loss.backward()
        opt.step()

    if args.checkpoint:
        torch.save(
            {"model": model.state_dict(), "config": cfg},
            args.checkpoint,
        )


class _DummyEncoder:
    """Matches HfQueryEncoder.encode API for offline tests."""

    def __init__(self, dim: int, device: torch.device) -> None:
        self._dim = dim
        self._device = device

    def encode(self, texts: list[str]) -> torch.Tensor:
        del texts
        return torch.randn(1, self._dim, device=self._device)


if __name__ == "__main__":
    main()
