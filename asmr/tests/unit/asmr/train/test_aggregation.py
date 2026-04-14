"""Unit tests for field aggregation modules."""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from asmr.train.aggregation import AggregationHead, MFARFieldAdapter
from asmr.train.features import build_aux_features
from asmr.train.inference import apply_aggregation_head
from asmr.train.losses import listwise_logit_loss, pairwise_hinge_loss


def test_mfar_adapter_matches_manual_weighting() -> None:
    torch.manual_seed(0)
    b, f_num, m_num, d_num = 2, 2, 2, 5
    h = 8
    q = torch.randn(b, h)
    scores = torch.randn(b, f_num, m_num, d_num)
    adapter = MFARFieldAdapter(h, f_num, m_num)
    out = adapter(q, scores)
    logits_fm = adapter._weight_logits(q)
    w = torch.softmax(logits_fm, dim=-1)
    flat = scores.view(b, f_num * m_num, d_num)
    expected = (w.unsqueeze(-1) * flat).sum(dim=1)
    assert torch.allclose(out, expected)


def test_aggregation_head_shape() -> None:
    b, f_num, m_num, d_num = 1, 2, 2, 4
    h, aux_dim = 8, 8
    q = torch.randn(b, h)
    scores = torch.randn(b, f_num, m_num, d_num)
    mask = torch.ones(b, f_num, d_num, dtype=torch.bool)
    aux = build_aux_features(scores, mask)
    head = AggregationHead(h, f_num, m_num, hidden_dim=16, aux_dim=aux_dim)
    logits = head(q, scores, aux)
    assert logits.shape == (b, d_num)


def test_listwise_loss_decreases_on_perfect_logits() -> None:
    logits = torch.tensor([[0.0, 10.0, 0.0]])
    rel = torch.tensor([[0.0, 1.0, 0.0]])
    loss = listwise_logit_loss(logits, rel, temperature=1.0)
    assert loss.item() < 0.01


def test_pairwise_hinge() -> None:
    logits = torch.tensor([[0.0, 1.0, 0.5]])
    rel = torch.tensor([[1.0, 0.0, 0.0]])
    loss = pairwise_hinge_loss(logits, rel, margin=0.1)
    assert loss.ndim == 0


def test_apply_aggregation_head_numpy() -> None:
    torch.manual_seed(1)
    f_num, m_num, d_num = 2, 2, 3
    h = 6
    scores = np.random.randn(f_num, m_num, d_num).astype(np.float32)
    mask = np.ones((f_num, d_num), dtype=bool)
    q = np.random.randn(h).astype(np.float32)
    doc_ids = ["10", "20", "30"]
    adapter = MFARFieldAdapter(h, f_num, m_num)
    out_ids, logits = apply_aggregation_head(
        doc_ids,
        scores,
        mask,
        q,
        adapter,
    )
    assert len(out_ids) == d_num
    assert logits.shape == (d_num,)
