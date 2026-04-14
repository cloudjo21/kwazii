"""Low-cost auxiliary features for aggregation (design doc section 5)."""

import torch
from torch import Tensor


def build_aux_features(
    scores: Tensor,
    field_mask: Tensor,
) -> Tensor:
    """Compute per-document statistics from field scores.

    Args:
        scores: [B, F, M, D] or [B, F, D]. If 4D, M scorers are mean-pooled
            for variance stats.
        field_mask: [B, F, D] True where field applies to document.

    Returns:
        Tensor [B, D, A] with A=8: var, max, min, mean, span, masked_mean,
        scorer_spread (if M>1 else 0), density.
    """
    if scores.dim() == 4:
        pooled = scores.mean(dim=2)
    else:
        pooled = scores

    b, f, d = pooled.shape
    masked = pooled.clone()
    masked = masked.masked_fill(~field_mask, float("nan"))

    nan_mask = torch.isnan(masked)
    mean_s = torch.nanmean(masked, dim=1).nan_to_num(0.0)
    diff_sq = (masked - mean_s.unsqueeze(1)).pow(2).nan_to_num(0.0)
    valid_count = (~nan_mask).float().sum(dim=1).clamp(min=1.0)
    var = diff_sq.sum(dim=1) / valid_count
    max_s = (
        masked.masked_fill(nan_mask, float("-inf")).max(dim=1).values.nan_to_num(0.0)
    )
    min_s = masked.masked_fill(nan_mask, float("inf")).min(dim=1).values.nan_to_num(0.0)
    span = max_s - min_s

    mask_count = field_mask.float().sum(dim=1).clamp(min=1.0)
    masked_sum = torch.where(
        field_mask,
        pooled,
        torch.zeros_like(pooled),
    ).sum(dim=1)
    masked_mean = masked_sum / mask_count

    if scores.dim() == 4 and scores.shape[2] > 1:
        spread = scores.max(dim=2).values - scores.min(dim=2).values
        spread = torch.where(
            field_mask,
            spread,
            torch.zeros_like(spread),
        ).mean(dim=1)
    else:
        spread = torch.zeros(b, d, device=scores.device, dtype=scores.dtype)

    density = mask_count / float(f)

    return torch.stack(
        [
            var,
            max_s,
            min_s,
            mean_s,
            span,
            masked_mean,
            spread,
            density,
            torch.ones(b, d, device=scores.device, dtype=scores.dtype),
        ],
        dim=-1,
    )


def normalize_scores_per_field_scorer(
    scores: Tensor,
    field_mask: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Batch-normalize scores per (field, scorer) across documents (Opt A)."""
    if scores.dim() == 3:
        scores_4 = scores.unsqueeze(2)
    else:
        scores_4 = scores
    m_expand = field_mask.unsqueeze(2).expand_as(scores_4).float()
    count = m_expand.sum(dim=-1, keepdim=True).clamp_min(1.0)
    masked_sum = torch.where(
        m_expand.bool(),
        scores_4,
        torch.zeros_like(scores_4),
    ).sum(dim=-1, keepdim=True)
    mean = masked_sum / count
    centered = scores_4 - mean
    centered = torch.where(m_expand.bool(), centered, torch.zeros_like(centered))
    var = (centered**2).sum(dim=-1, keepdim=True) / count
    std = (var + eps).sqrt()
    out = centered / std
    out = torch.where(m_expand.bool(), out, scores_4)
    return out.squeeze(2) if scores.dim() == 3 else out
