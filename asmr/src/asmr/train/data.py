"""Batch structures for ranking training."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RankingBatch:
    """One batch of queries with shared shortlist shape per design doc."""

    scores: Tensor
    """Shape [B, F, M, D] field x scorer x document scores."""

    field_mask: Tensor
    """Shape [B, F, D] True if document has field f."""

    query_emb: Tensor
    """Shape [B, H] query vectors."""

    relevance: Tensor
    """Shape [B, D] 1.0 for relevant document slot else 0.0."""

    aux: Tensor | None = None
    """Shape [B, D, A] optional precomputed aux; else built in forward."""
