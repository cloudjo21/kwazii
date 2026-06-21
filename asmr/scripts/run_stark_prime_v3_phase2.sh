#!/usr/bin/env bash
# FireCUDA / GPU host: STaRK-Prime MFARAll v3 Phase 2 full reproduction.
# Two-process pipeline: (A) streaming shortlist cache, (B) train + eval.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${ASMR_DATA_ROOT:-data/stark_prime}"
INDEX_DIR="${ASMR_INDEX_DIR:-data/stark_prime/index/prime}"
ENCODER="${ASMR_ENCODER:-contriever}"
HF_MODEL="${ASMR_HF_MODEL:-facebook/contriever-msmarco}"
OUT_DIR="${ASMR_OUT_DIR:-data/stark_prime/benchmark_results}"
CHUNK_SIZE="${ASMR_SHORTLIST_CHUNK_SIZE:-128}"
PARALLEL_FLAG=()
if [[ "${ASMR_V3_PARALLEL:-0}" == "1" ]]; then
  PARALLEL_FLAG=(--parallel)
fi

export ASMR_QUERY_CACHE=1
export ASMR_SHORTLIST_CACHE=1
export ASMR_REQUIRE_SHORTLIST_CACHE=1
export ASMR_V3_PARALLEL="${ASMR_V3_PARALLEL:-0}"
export ASMR_V3_MIGRATE_FAISS=0

echo "=== Phase 2 prep: query embedding caches ==="
uv run python -m asmr.evaluation.stark_prime_query_cache \
  --data-root "$DATA_ROOT" \
  --encoder "$ENCODER" \
  --splits train,test

echo "=== Phase 2 Step A: streaming train shortlist cache (separate process) ==="
uv run python -m asmr.evaluation.stark_prime_shortlist_cache \
  --data-root "$DATA_ROOT" \
  --index-dir "$INDEX_DIR" \
  --split train \
  --shortlist-k 100 \
  --encoder "$ENCODER" \
  --hf-model-name "$HF_MODEL" \
  --streaming \
  --chunk-size "$CHUNK_SIZE" \
  "${PARALLEL_FLAG[@]}"

echo "=== Phase 2 Step B: train + eval (require shortlist cache, defer index until eval) ==="
uv run python -m asmr.evaluation.stark_prime_benchmark_v3 \
  --data-root "$DATA_ROOT" \
  --index-dir "$INDEX_DIR" \
  --encoder "$ENCODER" \
  --hf-model-name "$HF_MODEL" \
  --training-phase 2 \
  --train-epochs 5 \
  --shortlist-k 100 \
  --shortlist-chunk-size "$CHUNK_SIZE" \
  --normalize-scores \
  --warm-query-cache \
  --use-shortlist-cache \
  --require-shortlist-cache \
  --no-parallel \
  --output "$OUT_DIR/mfar_all_v3_phase2_full.json"

echo "=== Done: $OUT_DIR/mfar_all_v3_phase2_full.json ==="
