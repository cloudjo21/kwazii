"""Tests for AggregationTrainer."""

import pytest

pytest.importorskip("torch")
import torch

from asmr.train.aggregation import MFARFieldAdapter
from asmr.train.config import TrainConfig
from asmr.train.data import RankingBatch
from asmr.train.trainer import AggregationTrainer


def test_trainer_step_runs() -> None:
    torch.manual_seed(0)
    b, f_num, m_num, d_num, h = 2, 2, 2, 5, 8
    scores = torch.randn(b, f_num, m_num, d_num)
    field_mask = torch.ones(b, f_num, d_num, dtype=torch.bool)
    query_emb = torch.randn(b, h)
    relevance = torch.zeros(b, d_num)
    relevance[:, 0] = 1.0
    batch = RankingBatch(
        scores=scores,
        field_mask=field_mask,
        query_emb=query_emb,
        relevance=relevance,
    )
    model = MFARFieldAdapter(h, f_num, m_num)
    cfg = TrainConfig(query_dim=h, temperature=0.1)
    trainer = AggregationTrainer(model, cfg)
    out = trainer.training_step(batch)
    assert out.loss.ndim == 0
    assert out.loss_rank.ndim == 0
