#!/usr/bin/env bash
set -eo pipefail
cd "${PROJECT_DIR:-$PWD}"
source slurm/env.sh
# All assets must already exist; no downloads or compilation on GPU workers.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export XFORMERS_DISABLED=1
PLAN="${1:?plan required}"
PHASE="${2:?probe or eval required}"
srun "$PYTHON_BIN" scripts/cluster_sr_reference_run.py \
  --plan "$PLAN" --phase "$PHASE" --index "${SLURM_ARRAY_TASK_ID:?array index required}"
