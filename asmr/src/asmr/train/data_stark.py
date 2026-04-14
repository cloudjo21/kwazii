"""STaRK-oriented dataset and batching (§13.3)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import Dataset

from asmr.train.data import RankingBatch

logger = logging.getLogger(__name__)


class SupportsQueryEncode(Protocol):
    """Anything with ``encode(texts) -> [B,H]`` (e.g. HfQueryEncoder)."""

    def encode(self, texts: list[str]) -> Tensor: ...


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

    @classmethod
    def from_jsonl(cls, path: Path) -> "StarkRankingDataset":
        """Load examples from a JSONL file.

        Each line is a JSON object with fields:
            query_text (str), doc_ids (list[str]),
            scores (list[list[list[float]]] — shape [F,M,D] or [F,D]),
            relevance (list[float] — length D),
            field_mask (list[list[bool]] — shape [F,D], optional).

        Args:
            path: Path to .jsonl file.

        Returns:
            StarkRankingDataset loaded into memory.
        """
        examples: list[StarkRankingExample] = []
        with open(path) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {lineno}: {e}") from e

                scores_raw = row["scores"]
                scores = np.array(scores_raw, dtype=np.float32)
                relevance = np.array(row["relevance"], dtype=np.float32)
                field_mask: npt.NDArray[np.bool_] | None = None
                if "field_mask" in row:
                    field_mask = np.array(row["field_mask"], dtype=bool)

                examples.append(
                    StarkRankingExample(
                        query_text=str(row["query_text"]),
                        doc_ids=[str(d) for d in row["doc_ids"]],
                        scores=scores,
                        relevance=relevance,
                        field_mask=field_mask,
                    )
                )
        logger.info("Loaded %d examples from %s", len(examples), path)
        return cls(examples)


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
