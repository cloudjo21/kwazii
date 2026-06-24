#!/usr/bin/env bash
# STaRK-MAG MFARAll benchmark (Phase 1: head-only).
# Smoke by default (full corpus + small query count). Set ASMR_BENCHMARK_FULL=1 for full run.
#
# Usage:
#   bash run_stark_mag_benchmark.sh                         # reuse existing index (error if not built)
#   bash run_stark_mag_benchmark.sh --build-index           # build index if missing, then run
#   bash run_stark_mag_benchmark.sh --rebuild-index         # force-rebuild index, then run
#   bash run_stark_mag_benchmark.sh --mps-thread-pct 30    # limit GPU SM usage to 30% via CUDA MPS
#
# Anti-pattern avoided: do NOT truncate corpus (max_docs=-1 always).
# A truncated corpus drops ground-truth docs -> Hit@k=0 misread as "broken pipeline".
set -euo pipefail

export PATH="/home/ed/.local/bin:$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."

# --- parse optional args ---
BUILD_INDEX_FLAG=()
WARM_QUERY_CACHE_FLAG=()
USE_QUERY_CACHE_FLAG=()
MPS_THREAD_PCT="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-50}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-index)       BUILD_INDEX_FLAG=(--build-index); shift ;;
    --rebuild-index)     BUILD_INDEX_FLAG=(--rebuild-index); shift ;;
    --warm-query-cache)  WARM_QUERY_CACHE_FLAG=(--warm-query-cache); shift ;;
    --use-query-cache)   USE_QUERY_CACHE_FLAG=(--use-query-cache); shift ;;
    --mps-thread-pct)    MPS_THREAD_PCT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

DATA_ROOT="${ASMR_DATA_ROOT:-data/stark_mag}"
ENCODER="${ASMR_ENCODER:-facebook/contriever-msmarco}"
CACHE_DIR="${ASMR_CACHE_DIR:-data/stark_mag/cache}"
OUT_DIR="${ASMR_OUT_DIR:-data/stark_mag/benchmark_results}"
# MPS_THREAD_PCT: resolved from --mps-thread-pct arg or CUDA_MPS_ACTIVE_THREAD_PERCENTAGE env.

if [[ "${ASMR_BENCHMARK_FULL:-0}" == "1" ]]; then
  TAG="full"
  MAX_TRAIN=-1
  MAX_EVAL=-1
  SHORTLIST_K=100
  EPOCHS=5
else
  TAG="smoke"
  MAX_TRAIN=200
  MAX_EVAL=50
  SHORTLIST_K=30
  EPOCHS=1
fi

echo "=== STaRK-MAG MFARAll benchmark [${TAG}] ==="
echo "    data_root : $DATA_ROOT"
echo "    encoder   : $ENCODER"
echo "    cache_dir : $CACHE_DIR"
echo "    shortlist_k=$SHORTLIST_K  epochs=$EPOCHS  max_train=$MAX_TRAIN  max_eval=$MAX_EVAL"
[[ ${#BUILD_INDEX_FLAG[@]} -gt 0 ]]      && echo "    index     : ${BUILD_INDEX_FLAG[*]}"
[[ ${#WARM_QUERY_CACHE_FLAG[@]} -gt 0 ]] && echo "    query cache: warm (build if missing)"
[[ ${#USE_QUERY_CACHE_FLAG[@]} -gt 0 ]]  && echo "    query cache: use existing"

uv run python -m asmr.evaluation.stark_mag_benchmark \
  --data-root "$DATA_ROOT" \
  --encoder "$ENCODER" \
  --shortlist-k "$SHORTLIST_K" \
  --train-epochs "$EPOCHS" \
  --max-docs -1 \
  --max-train-queries "$MAX_TRAIN" \
  --max-eval-queries "$MAX_EVAL" \
  --eval-k 20 \
  --cache-dir "$CACHE_DIR" \
  --mps-thread-pct "$MPS_THREAD_PCT" \
  "${BUILD_INDEX_FLAG[@]}" \
  "${WARM_QUERY_CACHE_FLAG[@]}" \
  "${USE_QUERY_CACHE_FLAG[@]}" \
  --output "$OUT_DIR/mfar_all_${TAG}.json"

echo "=== Done: $OUT_DIR/mfar_all_${TAG}.json ==="
