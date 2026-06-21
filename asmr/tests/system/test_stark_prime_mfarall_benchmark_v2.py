"""System test: STaRK-Prime MFARAll v2 (disk-backed indexes).

Requires:
  - asmr/data/stark_prime (HF snap-stanford/stark Prime SKB + QA)
  - CUDA + contriever-msmarco
  - ASMR_BENCHMARK_TESTS=1 for full suite
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "stark_prime"
_INDEX_DIR = _DATA_ROOT / "index" / "prime"
_FULL = os.getenv("ASMR_BENCHMARK_TESTS", "0") == "1"
_SKIP = not (_DATA_ROOT / "skb" / "processed" / "node_info.pkl").exists()


@pytest.mark.skipif(_SKIP, reason=f"STaRK-Prime data missing at {_DATA_ROOT}")
@pytest.mark.system
class TestStarkPrimeMfarAllBenchmarkV2:
    """GPU system test for STaRK-Prime + MFARAll v2 protocol."""

    def test_mfar_all_v2_smoke_or_full(self) -> None:
        from asmr.evaluation.stark_prime_benchmark_v2 import run_benchmark_v2

        report = run_benchmark_v2(
            _DATA_ROOT,
            _INDEX_DIR,
            shortlist_k=50 if _FULL else 20,
            train_epochs=3 if _FULL else 1,
            max_train_queries=-1 if _FULL else 200,
            max_eval_queries=-1 if _FULL else 50,
            eval_k=20,
            rebuild_index=os.getenv("ASMR_INDEX_REBUILD", "0") == "1",
        )
        m = report.metrics
        assert m.num_queries > 0
        assert 0.0 <= m.hit_at_1 <= 1.0
        assert 0.0 <= m.recall_at_20 <= 1.0
        assert 0.0 <= m.mrr <= 1.0
        if not _FULL:
            assert m.hit_at_1 > 0.0, "smoke must use full corpus; Hit@1=0 means misconfig"

        out_dir = _DATA_ROOT / "benchmark_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "full" if _FULL else "smoke"
        out_path = out_dir / f"mfar_all_v2_{tag}.json"
        out_path.write_text(
            json.dumps(
                {
                    "protocol": report.protocol,
                    "metrics": {
                        "hit_at_1": m.hit_at_1,
                        "recall_at_20": m.recall_at_20,
                        "mrr": m.mrr,
                        "num_queries": m.num_queries,
                        "elapsed_sec": m.elapsed_sec,
                    },
                    "ratio_vs_mfar_mfarall": report.ratio_vs_mfar_mfarall,
                    "notes": report.notes,
                },
                indent=2,
            )
        )
