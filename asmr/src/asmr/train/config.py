"""Training configuration for field aggregation."""

from dataclasses import dataclass
from enum import IntEnum


class TrainingPhase(IntEnum):
    """Phase 1 = head-only; Phase 2 = encoder + head joint."""

    HEAD_ONLY = 1
    JOINT = 2


@dataclass(frozen=True)
class TrainMfarConfig:
    """Hyperparameters for STaRK-Prime MFAR training."""

    shortlist_k: int = 100
    epochs: int = 5
    adapter_lr: float = 1e-3
    encoder_lr: float = 2e-5
    phase: TrainingPhase = TrainingPhase.HEAD_ONLY
    normalize_scores: bool = False
    weight_decay: float = 1e-2


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for aggregation head and losses."""

    query_dim: int = 768
    hidden_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.07
    lambda_rank: float = 1.0
    lambda_field: float = 0.0
    lambda_distill: float = 0.0
    use_aux_features: bool = True
    aux_dim: int = 9  # build_aux_features returns 9 values: var/max/min/mean/span/masked_mean/scorer_spread/density/bias
    normalize_scores: bool = (
        False  # apply normalize_scores_per_field_scorer before head (Opt A)
    )
