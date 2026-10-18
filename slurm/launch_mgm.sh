#!/usr/bin/env bash
# Two independent recipes, each capped at two concurrent one-GPU jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--submit" ) ]]; then
  echo "Usage: bash slurm/launch_mgm.sh [--submit]" >&2
  exit 2
fi
MGM_RUNS_ROOT="${MGM_RUNS_ROOT:-cluster_runs}"
MGM_DATA_ROOT="${MGM_DATA_ROOT:-data/div2k_multicrop128}"
MGM_PYTHON="${PYTHON_BIN:-python}"
COMMON=(
  --data-root "$MGM_DATA_ROOT" --data-protocol random128-multicrop-v4
  --train-crops-per-image 4 --eval-crops-per-image 5
  --width 120 --seeds 0 1 2
  --warmup-epochs 50 --field-warmup-epochs 50 --train-epochs 80
  --field-initialization warmup_features
  --mgm-optimizer authors --mgm-beta1 0.8
  --mgm-microbatch-size 32 --grad-accumulation 1
  --warmup-time-limit-minutes 60 --field-warmup-time-limit-minutes 90
  --train-time-limit-minutes 240 --eval-minutes 30 --max-parallel 2
)
"$MGM_PYTHON" scripts/cluster_submit.py \
  --run-root "$MGM_RUNS_ROOT/sr_v1_multi4_eval5_w50_f50_g80_b32_beta08" \
  --architecture-version 1 --variants mgm "${COMMON[@]}" "$@"
"$MGM_PYTHON" scripts/cluster_submit.py \
  --run-root "$MGM_RUNS_ROOT/sr_v2_multi4_eval5_w50_f50_g80_b32_beta08" \
  --architecture-version 2 --variants mgm_critic_noise "${COMMON[@]}" "$@"
