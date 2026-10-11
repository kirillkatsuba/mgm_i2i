#!/usr/bin/env bash
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23
#SBATCH --signal=USR1@120
#SBATCH --output=slurm/logs/%x-%A_%a.out

# Submission helper sets --chdir to the checkout; Slurm copies this script to spool.
set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
source slurm/env.sh
PLAN="${1:?plan path required}"
PHASE="${2:?warmup or train required}"
shift 2
# USR1 is sent to job steps; Python finishes its current update and saves atomically.
# Optional remaining arguments: --resume / --evaluate-only / --split test.
srun "$PYTHON_BIN" scripts/cluster_run.py --plan "$PLAN" --phase "$PHASE" \
    --index "${SLURM_ARRAY_TASK_ID:?array index required}" "$@"
