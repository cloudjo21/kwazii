"""System test: STaRK-Prime MFARAll end-to-end benchmark on GPU host.

Requires:
  - asmr/data/stark_prime (HF snap-stanford/stark Prime SKB + QA)
  - CUDA + contriever-msmarco download
  - ASMR_BENCHMARK_TESTS=1 to run full suite (otherwise smoke subset)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "stark_prime"
_FULL = os.getenv("ASMR_BENCHMARK_TESTS", "0") == "1"
_SKIP = not (_DATA_ROOT / "skb" / "processed" / "node_info.pkl").exists()


@pytest.mark.skipif(_SKIP, reason=f"STaRK-Prime data missing at {_DATA_ROOT}")
@pytest.mark.system
class TestStarkPrimeMfarAllBenchmark:
    """GPU system test for STaRK-Prime + MFARAll protocol."""

    def test_mfar_all_benchmark_smoke_or_full(self) -> None:
        from asmr.evaluation.stark_prime_benchmark import run_benchmark

        report = run_benchmark(
            _DATA_ROOT,
            shortlist_k=50 if _FULL else 20,
            train_epochs=3 if _FULL else 1,
            max_docs=-1 if _FULL else 2000,
            max_train_queries=-1 if _FULL else 200,
            max_eval_queries=-1 if _FULL else 100,
            eval_k=20,
        )
        m = report.metrics
        assert m.num_queries > 0
        assert 0.0 <= m.hit_at_1 <= 1.0
        assert 0.0 <= m.recall_at_20 <= 1.0
        assert 0.0 <= m.mrr <= 1.0

        out_dir = _DATA_ROOT / "benchmark_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "full" if _FULL else "smoke"
        out_path = out_dir / f"mfar_all_{tag}.json"
        out_path.write_text(
            json.dumps(
                {
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
