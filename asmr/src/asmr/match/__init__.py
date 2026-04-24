"""Image ranking inference module.

Provides top-k image ranking given one structured text query and a list of
image URLs.  The encoder backend is injected at construction time, so swapping
the underlying retrieval model requires no changes to the caller.

Typical usage::

    from asmr.match import ImageRanker, RankerConfig

    ranker = ImageRanker(encoder=my_encoder, config=RankerConfig(top_k=5))
    results = await ranker.rank("red cotton t-shirt", image_urls)
    # results: list[RankResult] sorted by score descending
"""

from asmr.match.config import RankerConfig
from asmr.match.indexer import CandidateImageIndexer
from asmr.match.ranker import ImageRanker, RankResult

__all__ = [
    "CandidateImageIndexer",
    "ImageRanker",
    "RankResult",
    "RankerConfig",
]
