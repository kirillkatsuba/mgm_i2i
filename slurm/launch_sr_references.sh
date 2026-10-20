#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON_BIN:-python}" scripts/cluster_sr_reference_submit.py \
  --run-root "${SR_REFERENCE_RUN_ROOT:-cluster_runs/sr_refs_multi5_val_r1}" \
  --data-root "${MGM_DATA_ROOT:-data/div2k_multicrop128}" \
  --weights-root "${MGM_WEIGHTS_ROOT:-weights}" "$@"
