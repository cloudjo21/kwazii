"""Training configuration for field aggregation."""

from dataclasses import dataclass


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
    normalize_scores: bool = False  # apply normalize_scores_per_field_scorer before head (Opt A)
