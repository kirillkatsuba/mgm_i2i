#!/usr/bin/env bash
# Matched multicrop data; two independent plans with two GPUs each by default.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--submit" ) ]]; then
  echo "Usage: bash slurm/launch_baselines.sh [--submit]" >&2
  exit 2
fi
MGM_RUNS_ROOT="${MGM_RUNS_ROOT:-cluster_runs}"
MGM_DATA_ROOT="${MGM_DATA_ROOT:-data/div2k_multicrop128}"
MGM_WEIGHTS_ROOT="${MGM_WEIGHTS_ROOT:-weights}"
MGM_PYTHON="${PYTHON_BIN:-python}"
BASELINE_RUN_TAG="${BASELINE_RUN_TAG:-r2}"
BASELINE_MAX_PARALLEL="${BASELINE_MAX_PARALLEL:-2}"
if [[ ! "$BASELINE_RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "BASELINE_RUN_TAG must contain only letters, digits, underscores or hyphens." >&2
  exit 2
fi
if [[ ! "$BASELINE_MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "BASELINE_MAX_PARALLEL must be a positive integer (per plan)." >&2
  exit 2
fi
CLASSIC_RUN="$MGM_RUNS_ROOT/baselines_multi4_eval5_w50_g80_b32_$BASELINE_RUN_TAG"
BRIDGE_RUN="$MGM_RUNS_ROOT/i2sb_multi4_eval5_e130_b32_$BASELINE_RUN_TAG"
if [[ "${1:-}" == "--submit" ]]; then
  for RUN in "$CLASSIC_RUN" "$BRIDGE_RUN"; do
    if [[ -e "$RUN/submission.json" ]]; then
      echo "Submission already attempted: $RUN/submission.json" >&2
      echo "Inspect those job IDs before resuming; no jobs submitted by this invocation." >&2
      exit 1
    fi
  done
fi
COMMON=(
  --data-root "$MGM_DATA_ROOT" --weights-root "$MGM_WEIGHTS_ROOT"
  --data-protocol random128-multicrop-v4
  --train-crops-per-image 4 --eval-crops-per-image 5
  --width 120 --architecture-version 2 --seeds 0 1 2
  --batch-size 32 --eval-batch-size 1
  --warmup-epochs 50 --warmup-time-limit-minutes 60
  --eval-minutes 30 --max-parallel "$BASELINE_MAX_PARALLEL"
)
classic_plan() {
  "$MGM_PYTHON" scripts/cluster_submit.py "${COMMON[@]}" \
    --run-root "$CLASSIC_RUN" --variants mse mse_perceptual esrgan \
    --train-epochs 80 --train-time-limit-minutes 120 "$@"
}
bridge_plan() {
  "$MGM_PYTHON" scripts/cluster_submit.py "${COMMON[@]}" \
    --run-root "$BRIDGE_RUN" --variants i2sb \
    --train-epochs 130 --train-time-limit-minutes 180 --i2sb-nfe 1 4 16 64 "$@"
}
# Validate both immutable plans before submitting either one. Preparation/weight
# hashes are additionally verified by cluster_submit.py on each real submission.
classic_plan
bridge_plan
if [[ "${1:-}" == "--submit" ]]; then
  classic_plan --submit
  bridge_plan --submit
fi
