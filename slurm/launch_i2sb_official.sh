#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--submit" ) ]]; then
  echo "Usage: bash slurm/launch_i2sb_official.sh [--submit]" >&2
  exit 2
fi
"${PYTHON_BIN:-python}" scripts/cluster_i2sb_official_submit.py \
  --run-root "${I2SB_OFFICIAL_RUN_ROOT:-cluster_runs/i2sb_official_sr4x_pool_multi5_val_r1}" \
  --data-root "${MGM_DATA_ROOT:-data/div2k_multicrop128}" \
  --weights-root "${MGM_WEIGHTS_ROOT:-weights}" \
  --bundle "${I2SB_OFFICIAL_BUNDLE:-weights/i2sb_official_sr4x_pool/manifest.json}" \
  --upstream "${I2SB_OFFICIAL_UPSTREAM:-third_party/I2SB-official}" \
  --max-parallel "${I2SB_OFFICIAL_MAX_PARALLEL:-2}" "$@"
