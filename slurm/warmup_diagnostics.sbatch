#!/usr/bin/env bash
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
set -eo pipefail
cd "${PROJECT_DIR:-$PWD}"
source slurm/env.sh
SOURCE_RUN="${1:?source run directory required}"
OUTPUT_ROOT="${2:?diagnostics output directory required}"
SEED="${SLURM_ARRAY_TASK_ID:?array seed required}"
CHECKPOINT="$SOURCE_RUN/warmup/seed$SEED/latest.pt"
srun "$PYTHON_BIN" scripts/cluster_diagnose_warmup.py \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT_ROOT/seed$SEED"
if [[ "$SEED" == "0" ]]; then
  srun "$PYTHON_BIN" scripts/cluster_probe_mgm.py \
    --checkpoint "$CHECKPOINT" --output "$OUTPUT_ROOT/memory_probe.json" \
    --batch-sizes 32 --mgm-beta1 0.8
fi
