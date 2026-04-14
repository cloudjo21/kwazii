"""Ranking losses for shortlist training."""

import torch
import torch.nn.functional as F
from torch import Tensor


def listwise_logit_loss(
    logits: Tensor,
    relevance: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Softmax cross-entropy over shortlist when exactly one positive.

    Args:
        logits: [B, D]
        relevance: [B, D] non-negative; at least one positive per row expected

    Returns:
        Scalar loss.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    target = relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(scaled, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def pairwise_hinge_loss(
    logits: Tensor,
    relevance: Tensor,
    margin: float = 0.1,
) -> Tensor:
    """Hinge margin between highest positive and negatives.

    Args:
        logits: [B, D]
        relevance: [B, D] binary or weights

    Returns:
        Scalar loss.
    """
    pos_mask = relevance > 0.5
    if not pos_mask.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

    pos_scores = logits.masked_fill(~pos_mask, float("-inf"))
    pos_max, _ = pos_scores.max(dim=-1)

    neg_mask = ~pos_mask
    neg_scores = logits.masked_fill(~neg_mask, float("inf"))
    neg_min, _ = neg_scores.min(dim=-1)

    loss = F.relu(margin - (pos_max - neg_min))
    return loss.mean()
