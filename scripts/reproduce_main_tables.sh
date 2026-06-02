#!/usr/bin/env bash
# Regenerate main cliff tables (1-5) from released results. No retraining.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
"$PY" "$HERE/artifact/reproduce_main_tables.py"
echo "Output: results/reproduced_tables/"
