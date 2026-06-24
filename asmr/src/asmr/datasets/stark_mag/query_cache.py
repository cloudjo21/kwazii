"""STaRK-MAG query cache — thin wrapper over asmr.datasets.query_cache."""

import argparse
import logging
from pathlib import Path

from asmr.datasets.query_cache import (
    QueryEmbeddingCache,
    build_query_emb_cache as _build_shared,
    is_query_cache_built,
    load_query_caches,
    resolve_query_emb,
)
from asmr.encode.protocol import NamedTextEncoderProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "QueryEmbeddingCache",
    "build_mag_query_emb_cache",
    "is_query_cache_built",
    "load_query_caches",
    "resolve_query_emb",
]


def build_mag_query_emb_cache(
    data_root: Path,
    split: str,
    encoder: NamedTextEncoderProtocol,
    cache_dir: Path,
    *,
    batch_size: int = 64,
    force: bool = False,
) -> tuple[Path, Path]:
    """MAG-specific builder: loads MAG queries then delegates to shared cache builder."""
    from asmr.datasets.stark_mag.loader import load_mag_queries

    queries = load_mag_queries(data_root, split)
    return _build_shared(
        queries, encoder, cache_dir, split, batch_size=batch_size, force=force
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build STaRK-MAG query embedding cache",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/stark_mag"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/stark_mag/cache"))
    parser.add_argument(
        "--splits", default="train,test", help="Comma-separated splits to cache"
    )
    parser.add_argument("--encoder", default="facebook/contriever-msmarco")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from asmr.train.query_encoder import HfQueryEncoder

    encoder = HfQueryEncoder(model_name=args.encoder)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    for split in splits:
        build_mag_query_emb_cache(
            args.data_root,
            split,
            encoder,
            args.cache_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
