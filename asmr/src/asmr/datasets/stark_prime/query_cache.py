"""Query embedding cache for STaRK-Prime MFARAll."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from asmr.datasets.stark_prime.loader import PrimeQuery, load_prime_queries
from asmr.encode.protocol import NamedTextEncoderProtocol
from asmr.evaluation.query_encoders import create_query_encoder
from fde.config import PromptType

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, split: str, encoder_name: str) -> Path:
    safe = encoder_name.replace("/", "_").replace(" ", "_")
    tagged = cache_dir / f"query_emb_{split}_{safe}.pt"
    if tagged.exists():
        return tagged
    legacy = cache_dir / f"query_emb_{split}.pt"
    return legacy


def build_query_emb_cache(
    data_root: Path,
    split: str,
    encoder: NamedTextEncoderProtocol,
    cache_dir: Path,
    *,
    batch_size: int = 64,
) -> Path:
    """Encode all queries in a split and persist embeddings."""
    queries = load_prime_queries(data_root, split)
    texts = [q.query for q in queries]
    ids = [q.query_id for q in queries]
    logger.info("Encoding %d %s queries with %s", len(queries), split, encoder.name)

    all_emb: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        emb = encoder.encode_text(
            batch,
            PromptType.QUERY,
            batch_size=batch_size,
        )
        all_emb.append(emb)
    stacked = np.concatenate(all_emb, axis=0)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"query_emb_{split}_{encoder.name.replace('/', '_')}.pt"
    torch.save(
        {"ids": ids, "emb": torch.from_numpy(stacked), "encoder": encoder.name},
        path,
    )
    logger.info("Wrote query cache %s", path)
    return path


class QueryEmbeddingCache:
    """O(1) lookup of precomputed query embeddings by query_id."""

    def __init__(self, cache_path: Path) -> None:
        data = torch.load(cache_path, weights_only=True)
        ids: list[int] = list(data["ids"])
        emb_tensor = data["emb"]
        if isinstance(emb_tensor, torch.Tensor):
            emb = emb_tensor.numpy()
        else:
            emb = np.asarray(emb_tensor)
        self._id_to_emb: dict[int, np.ndarray] = {
            qid: emb[i] for i, qid in enumerate(ids)
        }

    def get(self, query_id: int) -> np.ndarray | None:
        return self._id_to_emb.get(query_id)

    def get_or_encode(
        self,
        query: PrimeQuery,
        encoder: NamedTextEncoderProtocol,
    ) -> np.ndarray:
        cached = self.get(query.query_id)
        if cached is not None:
            return cached
        return encoder.encode_text(
            [query.query],
            PromptType.QUERY,
        )[0]


def resolve_query_emb(
    query: PrimeQuery,
    encoder: NamedTextEncoderProtocol,
    caches: dict[str, QueryEmbeddingCache],
    split: str,
) -> np.ndarray:
    """Return query embedding from cache or on-the-fly encode."""
    cache = caches.get(split)
    if cache is not None:
        return cache.get_or_encode(query, encoder)
    return encoder.encode_text([query.query], PromptType.QUERY)[0]


def load_query_caches(
    cache_dir: Path,
    splits: tuple[str, ...] = ("train", "test"),
    *,
    encoder_name: str | None = None,
) -> dict[str, QueryEmbeddingCache]:
    """Load split caches that exist on disk."""
    caches: dict[str, QueryEmbeddingCache] = {}
    for split in splits:
        candidates: list[Path] = []
        if encoder_name is not None:
            safe = encoder_name.replace("/", "_").replace(" ", "_")
            candidates.append(cache_dir / f"query_emb_{split}_{safe}.pt")
        candidates.append(cache_dir / f"query_emb_{split}.pt")
        path = next((p for p in candidates if p.exists()), None)
        if path is not None:
            caches[split] = QueryEmbeddingCache(path)
            logger.info("Loaded query cache %s", path)
    return caches


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build STaRK-Prime query embedding cache",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/stark_prime"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/stark_prime/cache"),
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated splits to cache",
    )
    parser.add_argument("--encoder", default="jinavera")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    encoder = create_query_encoder(args.encoder)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    for split in splits:
        build_query_emb_cache(
            args.data_root,
            split,
            encoder,
            args.cache_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
