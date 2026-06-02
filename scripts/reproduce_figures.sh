#!/usr/bin/env bash
# Regenerate main cliff figures (2,3,4) from released results into figures/reproduced/.
# Requires matplotlib + numpy. Authoritative figures are in figures/main and figures/appendix.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
"$PY" "$HERE/artifact/reproduce_figures.py"
