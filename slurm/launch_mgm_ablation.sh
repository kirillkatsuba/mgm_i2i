#!/usr/bin/env bash
# A single plan caps all v1/v2 jobs together; no host-memory request.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON_BIN:-python}" scripts/cluster_mgm_ablate.py "$@"
