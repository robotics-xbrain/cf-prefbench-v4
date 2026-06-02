#!/usr/bin/env bash
# Structural integrity check. Writes verify_artifact_results.json.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
"$PY" "$HERE/artifact/verify_artifact.py"
