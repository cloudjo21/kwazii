"""STaRK-oriented dataset and batching (§13.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import Dataset

from asmr.train.data import RankingBatch


class SupportsQueryEncode(Protocol):
    """Anything with ``encode(texts) -> [B,H]`` (e.g. HfQueryEncoder)."""

    def encode(self, texts: list[str]) -> Tensor:
        ...


@dataclass
class StarkRankingExample:
    """One training query with a fixed shortlist (§13.3)."""

    query_text: str
    doc_ids: list[str]
    scores: npt.NDArray[np.float32]
    """Shape [F, M, D] or [F, D]."""

    relevance: npt.NDArray[np.float32]
    """Shape [D], multi-label allowed."""

    field_mask: npt.NDArray[np.bool_] | None = None
    """Shape [F, D]; default all True."""


class StarkRankingDataset(Dataset[StarkRankingExample]):
    """In-memory list of examples (load JSONL upstream)."""

    def __init__(self, examples: list[StarkRankingExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> StarkRankingExample:
        return self._examples[idx]


def collate_stark_batch(
    batch: list[StarkRankingExample],
    query_encoder: SupportsQueryEncode,
    device: torch.device,
) -> RankingBatch:
    """Batch size 1 only (variable shortlist width D)."""
    if len(batch) != 1:
        msg = "collate_stark_batch currently supports batch size 1"
        raise NotImplementedError(msg)
    ex = batch[0]
    scores_np = ex.scores
    if scores_np.ndim == 2:
        scores_t = torch.from_numpy(scores_np).float().unsqueeze(1)
    elif scores_np.ndim == 3:
        scores_t = torch.from_numpy(scores_np).float()
    else:
        raise ValueError("scores must be [F,D] or [F,M,D]")

    scores_t = scores_t.unsqueeze(0).to(device)
    f_num = int(scores_t.shape[1])
    d_num = int(scores_t.shape[-1])

    if ex.field_mask is not None:
        fm = torch.from_numpy(np.asarray(ex.field_mask)).bool()
    else:
        fm = torch.ones((f_num, d_num), dtype=torch.bool)
    field_mask = fm.unsqueeze(0).to(device)

    rel = torch.from_numpy(np.asarray(ex.relevance)).float().unsqueeze(0).to(device)

    q_emb = query_encoder.encode([ex.query_text]).to(device)

    return RankingBatch(
        scores=scores_t,
        field_mask=field_mask,
        query_emb=q_emb,
        relevance=rel,
        aux=None,
    )
