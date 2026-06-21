"""Pickle-based shortlist cache for STaRK-MAG training examples."""

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from asmr.datasets.stark_prime.torch_dataset import StarkRankingExample

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1


def _encoder_slug(encoder_name: str) -> str:
    return encoder_name.replace("/", "_").replace(" ", "_")


def shortlist_cache_path(
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> Path:
    slug = _encoder_slug(encoder_name)
    return cache_dir / "mag_shortlists" / f"{split}_k{shortlist_k}_{slug}.pkl"


def save_shortlist_cache(
    examples: "list[StarkRankingExample]",
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> None:
    """Serialize examples to a pickle file under cache_dir."""
    path = shortlist_cache_path(cache_dir, split, shortlist_k, encoder_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"version": _CACHE_VERSION, "examples": examples}
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved %d MAG shortlist examples to %s", len(examples), path)


def load_shortlist_cache(
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> "Optional[list[StarkRankingExample]]":
    """Load examples from pickle cache. Returns None on miss or version mismatch."""
    path = shortlist_cache_path(cache_dir, split, shortlist_k, encoder_name)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload: dict[str, Any] = pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load MAG shortlist cache %s: %s", path, exc)
        return None
    if payload.get("version") != _CACHE_VERSION:
        logger.warning(
            "MAG shortlist cache version mismatch (expected %d, got %s); ignoring",
            _CACHE_VERSION,
            payload.get("version"),
        )
        return None
    examples = payload["examples"]
    logger.info("Loaded %d MAG shortlist examples from %s", len(examples), path)
    return examples
