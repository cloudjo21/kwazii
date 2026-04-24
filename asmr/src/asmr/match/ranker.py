"""ImageRanker — top-k image ranking from text query and candidate URLs.

Wraps DocumentRetriever.retrieve_with_aggregation_head to provide a clean,
high-level API that accepts raw text and image URLs and returns ranked results.

Design invariants:
- Encoder is injected: swap BaseFdeEncoder subclass to change the model.
- All torch/fde imports are lazy (inside function bodies).
- rank_batch() runs queries concurrently with asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from asmr.match.config import RankerConfig
from asmr.match.indexer import CandidateImageIndexer

if TYPE_CHECKING:
    from fde.base import BaseFdeEncoder

logger = logging.getLogger(__name__)


@dataclass
class RankResult:
    """Single image ranking result.

    Attributes:
        url: Original image URL or path.
        rank: 1-based rank position (1 = best match).
        score: Relevance score (higher is better).
    """

    url: str
    rank: int
    score: float


def _aggregate_scores(
    scores: np.ndarray,
) -> np.ndarray:
    """Collapse a [F, D] or [F, M, D] score matrix to a 1-D score vector.

    Args:
        scores: Score array from retrieve_with_aggregation_head (no-head path).

    Returns:
        1-D float32 array of length D.
    """
    if scores.ndim == 3:
        return scores.max(axis=(0, 1))
    if scores.ndim == 2:
        return scores.max(axis=0)
    return scores.flatten()


class ImageRanker:
    """Ranks candidate image URLs against a structured text query.

    The ranker builds an ephemeral FAISS index from the candidate images on
    every call to rank().  For ≤20 images this is fast enough for real-time
    use.  Use rank_batch() to process multiple queries concurrently.

    Attributes:
        encoder: FDE encoder (swappable — any BaseFdeEncoder subclass).
        config: Ranker configuration.
    """

    def __init__(
        self,
        encoder: BaseFdeEncoder,
        config: RankerConfig | None = None,
    ) -> None:
        self._encoder = encoder
        self._config = config or RankerConfig()
        self._indexer = CandidateImageIndexer(encoder, self._config)

    async def rank(
        self,
        text: str,
        image_urls: list[str],
        k: int | None = None,
        aggregation_head: Optional[Any] = None,
        query_encoder: Optional[Any] = None,
    ) -> list[RankResult]:
        """Rank image_urls by relevance to text.

        Args:
            text: Structured text query (non-empty).
            image_urls: Candidate image URLs or local paths (non-empty, max ~20).
            k: Number of top results to return.  Defaults to config.top_k.
                Clamped to len(image_urls) if larger.
            aggregation_head: Optional learned re-ranking head
                (MFARFieldAdapter or AggregationHead).
            query_encoder: Encoder for query embedding; required when
                aggregation_head is not None and no pre-computed embedding is
                available.

        Returns:
            List of RankResult sorted by score descending, length
            min(k, len(image_urls)).

        Raises:
            ValueError: If text is empty or image_urls is empty.
        """
        if not text:
            raise ValueError("text must be non-empty")
        if not image_urls:
            raise ValueError("image_urls must be non-empty")

        import torch as _torch

        from asmr.retrieve.query import Query

        top_k = min(k if k is not None else self._config.top_k, len(image_urls))
        retrieve_k = len(image_urls)

        retriever, doc_ids = self._indexer.build(image_urls)
        url_by_id: dict[str, str] = dict(zip(doc_ids, image_urls))

        query = Query.from_text(text)
        device = _torch.device(self._config.device)

        returned_ids, scores_arr = await retriever.retrieve_with_aggregation_head(
            query,
            k=retrieve_k,
            aggregation_head=aggregation_head,
            query_encoder=query_encoder,
            head_device=device,
        )

        scores_np = np.asarray(scores_arr, dtype=np.float32)

        if aggregation_head is None:
            flat = _aggregate_scores(scores_np)
            order = np.argsort(flat)[::-1][:top_k]
            ranked_ids = [returned_ids[i] for i in order]
            ranked_scores = flat[order]
        else:
            ranked_ids = list(returned_ids[:top_k])
            ranked_scores = scores_np.flatten()[:top_k]

        results: list[RankResult] = []
        for rank_idx, (doc_id, score) in enumerate(
            zip(ranked_ids, ranked_scores), start=1
        ):
            url = url_by_id.get(str(doc_id), str(doc_id))
            results.append(RankResult(url=url, rank=rank_idx, score=float(score)))

        return results

    async def rank_batch(
        self,
        queries: list[tuple[str, list[str]]],
        k: int | None = None,
        aggregation_head: Optional[Any] = None,
        query_encoder: Optional[Any] = None,
    ) -> list[list[RankResult]]:
        """Rank images for multiple queries concurrently.

        Args:
            queries: List of (text, image_urls) pairs.
            k: Top-k per query (same for all queries in the batch).
            aggregation_head: Optional shared re-ranking head.
            query_encoder: Encoder for query embeddings.

        Returns:
            List of rank result lists, one per query.
        """
        if not queries:
            return []

        tasks = [
            self.rank(
                text,
                urls,
                k=k,
                aggregation_head=aggregation_head,
                query_encoder=query_encoder,
            )
            for text, urls in queries
        ]
        return list(await asyncio.gather(*tasks))
