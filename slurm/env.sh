#!/usr/bin/env bash
# Sourced by batch scripts. Activate conda before nounset (conda may use unset vars).
set -eo pipefail
if [[ -n "${CONDA_ENV:-}" ]]; then
    if [[ -z "${CONDA_SH:-}" ]]; then
        CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
    fi
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN="${PYTHON_BIN:-python}"
