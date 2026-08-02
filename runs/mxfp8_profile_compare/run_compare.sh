#!/usr/bin/env bash
# Batch=4 correctness/debug suite of the mxfp8 profile compare.
#
# Thin wrapper over run_compare.py: each config runs in its own subprocess
# (P087 -- back-to-back configs in one process tree hit a cuBLAS internal
# error; see RESULTS.md "Methodology note") and the sweep is fail-fast.
#
# Usage:  ./runs/mxfp8_profile_compare/run_compare.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${HERE}/run_compare.py" --suite b4 "$@"
