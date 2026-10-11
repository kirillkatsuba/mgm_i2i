#!/bin/sh
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links offline/wheels -r requirements.lock
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/check_offline.py
