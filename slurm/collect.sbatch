#!/usr/bin/env bash
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5
#SBATCH --output=slurm/logs/collect-%j.out
set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
source slurm/env.sh
PLAN="${1:?plan path required}"
shift
"$PYTHON_BIN" scripts/cluster_collect.py --plan "$PLAN" "$@"
