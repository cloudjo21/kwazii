"""Unit tests for v2 streaming STaRK-Prime shortlist cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asmr.datasets.stark_prime.loader import PrimeQuery
from asmr.datasets.stark_prime.shortlist_cache import (
    ShortlistCacheMissingError,
    StreamingShortlistWriter,
    StreamingStarkRankingDataset,
    load_shortlist_cache,
    load_shortlist_manifest,
    save_shortlist_cache,
    shortlist_cache_dir_v2,
)
from asmr.datasets.stark_prime.torch_dataset import StarkRankingExample
from asmr.train.stark_prime_training import (
    ShortlistCacheMissingError as TrainingCacheMissingError,
    resolve_train_dataset,
)


def _make_example(text: str = "query text") -> StarkRankingExample:
    return StarkRankingExample(
        query_text=text,
        doc_ids=["d1", "d2"],
        scores=np.ones((2, 2, 2), dtype=np.float32),
        relevance=np.array([1.0, 0.0], dtype=np.float32),
        field_mask=np.ones((2, 2), dtype=bool),
    )


class TestStreamingShortlistWriter:
    def test_chunked_roundtrip(self, tmp_path: Path) -> None:
        """Writer flushes multiple chunks and manifest marks complete."""
        writer = StreamingShortlistWriter(
            tmp_path,
            "train",
            shortlist_k=50,
            encoder_name="enc_test",
            chunk_size=64,
        )
        for i in range(250):
            writer.append(_make_example(f"q{i}"), query_id=i)
        cache_dir = writer.finalize()

        manifest = load_shortlist_manifest(tmp_path, "train", 50, "enc_test")
        assert manifest is not None
        assert manifest.complete
        assert manifest.num_examples == 250
        assert manifest.num_chunks == 4
        assert cache_dir == shortlist_cache_dir_v2(tmp_path, "train", 50, "enc_test")

        loaded = load_shortlist_cache(tmp_path, "train", 50, "enc_test")
        assert loaded is not None
        assert len(loaded) == 250
        assert loaded[127].query_text == "q127"

    def test_incomplete_manifest_not_loadable(self, tmp_path: Path) -> None:
        """Incomplete manifest cannot back a streaming dataset."""
        writer = StreamingShortlistWriter(
            tmp_path,
            "train",
            10,
            "enc_test",
            chunk_size=64,
        )
        writer.append(_make_example(), query_id=1)
        writer._write_manifest(complete=False)  # noqa: SLF001

        manifest = load_shortlist_manifest(tmp_path, "train", 10, "enc_test")
        assert manifest is not None
        assert not manifest.complete
        with pytest.raises(ShortlistCacheMissingError):
            StreamingStarkRankingDataset(manifest)


class TestStreamingDataset:
    def test_chunk_boundary_random_access(self, tmp_path: Path) -> None:
        """Dataset resolves indices across chunk boundaries."""
        writer = StreamingShortlistWriter(
            tmp_path,
            "train",
            10,
            "enc_test",
            chunk_size=100,
        )
        for i in range(250):
            writer.append(_make_example(f"q{i}"), query_id=i)
        writer.finalize()
        manifest = load_shortlist_manifest(tmp_path, "train", 10, "enc_test")
        assert manifest is not None
        ds = StreamingStarkRankingDataset(manifest)
        assert ds[0].query_text == "q0"
        assert ds[127].query_text == "q127"
        assert ds[128].query_text == "q128"
        assert ds[249].query_text == "q249"


class TestLegacyV1Fallback:
    def test_v1_still_loads(self, tmp_path: Path) -> None:
        """Monolithic v1 cache remains readable."""
        ex = _make_example()
        queries = [PrimeQuery(query_id=1, query="query text", answer_ids=["d1"])]
        save_shortlist_cache(
            [ex],
            queries,
            tmp_path,
            "train",
            shortlist_k=10,
            encoder_name="enc_a",
        )
        loaded = load_shortlist_cache(tmp_path, "train", 10, "enc_a")
        assert loaded is not None
        assert len(loaded) == 1


class TestRequireShortlistCache:
    def test_raises_when_cache_missing(self, tmp_path: Path) -> None:
        """Required cache mode fails fast without inline shortlist."""
        encoder = type(
            "Enc",
            (),
            {"name": "enc_x", "supports_joint_training": lambda self: True},
        )()
        queries = [PrimeQuery(query_id=1, query="q", answer_ids=["d1"])]
        with pytest.raises(TrainingCacheMissingError):
            resolve_train_dataset(
                queries,
                None,
                encoder,  # type: ignore[arg-type]
                {},
                shortlist_k=10,
                parallel=False,
                cache_dir=tmp_path,
                require_shortlist_cache=True,
            )
