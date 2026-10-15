#!/usr/bin/env bash
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/full-%A_%a.out

set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
source slurm/env.sh
PLAN="${1:?path to existing plan.json required}"
shift
srun "$PYTHON_BIN" scripts/cluster_infer_full.py --plan "$PLAN" \
    --index "${SLURM_ARRAY_TASK_ID:?array index required}" "$@"
