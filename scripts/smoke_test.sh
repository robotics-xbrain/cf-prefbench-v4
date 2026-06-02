#!/usr/bin/env bash
# Smoke test for the CF-PrefBench v4 / LIB v0 artifact.
# Uses whatever `python` is on PATH (override with $PYTHON). Requires numpy;
# torch enables the dry-run forward pass. Writes smoke_test_results.json.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python3}"
echo "Using interpreter: $PY"
"$PY" "$HERE/artifact/smoke_test.py"
rc=$?
echo "smoke_test.sh exit code: $rc"
exit $rc
