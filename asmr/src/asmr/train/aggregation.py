"""Query-conditioned field aggregation: mFAR linear G and MLP G_theta."""

import math

import torch
from torch import Tensor, nn


class MFARFieldAdapter(nn.Module):
    """mFAR-style softmax weights over (field, scorer) channels.

    G(q,f,m) = softmax_{f,m}(a_{f,m}^T q). Document score is weighted sum
    of per-field scores (same weights for all documents in shortlist).
    """

    def __init__(
        self,
        query_dim: int,
        num_fields: int,
        num_scorers: int,
    ) -> None:
        super().__init__()
        self._num_fields = num_fields
        self._num_scorers = num_scorers
        fm = num_fields * num_scorers
        self._weight_logits = nn.Linear(query_dim, fm, bias=False)
        nn.init.normal_(self._weight_logits.weight, std=0.02)

    def forward(
        self,
        query_emb: Tensor,
        scores: Tensor,
    ) -> Tensor:
        """Compute document logits.

        Args:
            query_emb: [B, H]
            scores: [B, F, M, D] per-document field scores

        Returns:
            Logits [B, D]
        """
        if scores.dim() != 4:
            msg = f"scores must be [B,F,M,D], got {scores.shape}"
            raise ValueError(msg)
        b, f_num, m_num, d_num = scores.shape
        if f_num != self._num_fields or m_num != self._num_scorers:
            msg = "scores F,M must match constructor"
            raise ValueError(msg)

        logits_fm = self._weight_logits(query_emb)
        w = torch.softmax(logits_fm, dim=-1)
        flat = scores.view(b, f_num * m_num, d_num)
        return (w.unsqueeze(-1) * flat).sum(dim=1)


class AggregationHead(nn.Module):
    """Small MLP over flattened field scores + query + aux (G_theta)."""

    def __init__(
        self,
        query_dim: int,
        num_fields: int,
        num_scorers: int,
        hidden_dim: int,
        aux_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._num_fields = num_fields
        self._num_scorers = num_scorers
        in_dim = num_fields * num_scorers + query_dim + aux_dim
        self._mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(hidden_dim)))

    def forward(
        self,
        query_emb: Tensor,
        scores: Tensor,
        aux: Tensor,
    ) -> Tensor:
        """Per-document logits.

        Args:
            query_emb: [B, H]
            scores: [B, F, M, D]
            aux: [B, D, A]

        Returns:
            [B, D]
        """
        if scores.dim() != 4:
            raise ValueError("scores must be [B,F,M,D]")
        b, f_num, m_num, d_num = scores.shape
        flat = scores.permute(0, 3, 1, 2).reshape(b, d_num, f_num * m_num)
        q = query_emb.unsqueeze(1).expand(-1, d_num, -1)
        x = torch.cat([flat, q, aux], dim=-1)
        out = self._mlp(x).squeeze(-1) * self._scale
        return out
