"""Minimal training step helper; full loop left to Lightning or script."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from asmr.train.aggregation import AggregationHead, MFARFieldAdapter
from asmr.train.config import TrainConfig
from asmr.train.data import RankingBatch
from asmr.train.features import build_aux_features, normalize_scores_per_field_scorer
from asmr.train.losses import listwise_logit_loss


@dataclass
class StepOutput:
    """Loss bundle for one optimizer step."""

    loss: Tensor
    loss_rank: Tensor


class AggregationTrainer:
    """Wraps forward + loss for a single ranking batch."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
    ) -> None:
        self._model = model
        self._config = config

    def training_step(self, batch: RankingBatch) -> StepOutput:
        """Compute loss for one batch."""
        scores = batch.scores
        if self._config.normalize_scores:
            scores = normalize_scores_per_field_scorer(scores, batch.field_mask)

        aux: Tensor | None = batch.aux
        if isinstance(self._model, MFARFieldAdapter):
            logits = self._model(batch.query_emb, scores)
        elif isinstance(self._model, AggregationHead):
            if aux is None and self._config.use_aux_features:
                aux = build_aux_features(scores, batch.field_mask)
            if aux is None:
                b = batch.scores.shape[0]
                d_num = batch.scores.shape[-1]
                aux = torch.zeros(
                    b,
                    d_num,
                    self._config.aux_dim,
                    device=batch.scores.device,
                    dtype=batch.scores.dtype,
                )
            logits = self._model(
                batch.query_emb,
                scores,
                aux,
            )
        else:
            msg = f"Unsupported model: {type(self._model)}"
            raise TypeError(msg)

        loss_rank = listwise_logit_loss(
            logits,
            batch.relevance,
            temperature=self._config.temperature,
        )
        loss = self._config.lambda_rank * loss_rank
        return StepOutput(loss=loss, loss_rank=loss_rank)
