#!/usr/bin/env bash
# Batch=16 compute-density suite: bf16 vs mxfp8_gemm_ready.
#
# Thin wrapper over run_compare.py: each config runs in its own subprocess so
# JIT/cuBLAS state cannot leak across runs (we hit a cuBLAS internal error in
# the back-to-back batch=4 sweep), and the sweep is fail-fast (P087).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${HERE}/run_compare.py" --suite b16 "$@"
