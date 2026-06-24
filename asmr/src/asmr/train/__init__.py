"""Trainable multi-field aggregation (mFAR-style + composition head).

Requires PyTorch. Install with the extra ``train`` (``pip install asmr[train]``)
or use the repo root ``kwazii`` environment which already depends on torch.
"""

from asmr.train.aggregation import AggregationHead, MFARFieldAdapter
from asmr.train.config import TrainConfig
from asmr.train.data import RankingBatch
from asmr.train.features import build_aux_features, normalize_scores_per_field_scorer
from asmr.train.gpu_setup import setup_gpu
from asmr.train.inference import apply_aggregation_head
from asmr.train.losses import listwise_logit_loss, pairwise_hinge_loss
from asmr.train.query_encoder import HfQueryEncoder
from asmr.train.trainer import AggregationTrainer, StepOutput

__all__ = [
    "AggregationHead",
    "AggregationTrainer",
    "HfQueryEncoder",
    "MFARFieldAdapter",
    "RankingBatch",
    "StepOutput",
    "TrainConfig",
    "apply_aggregation_head",
    "build_aux_features",
    "collate_stark_batch",
    "listwise_logit_loss",
    "normalize_scores_per_field_scorer",
    "pairwise_hinge_loss",
    "setup_gpu",
]
