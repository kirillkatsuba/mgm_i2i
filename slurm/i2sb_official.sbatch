#!/usr/bin/env bash
# Resource requests and dependencies come from the signed evaluation plan.
set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
source slurm/env.sh
PLAN="${1:?plan path required}"
PHASE="${2:?probe or eval required}"
srun "$PYTHON_BIN" scripts/cluster_i2sb_official_run.py \
  --plan "$PLAN" --phase "$PHASE" --index "${SLURM_ARRAY_TASK_ID:?array index required}"
