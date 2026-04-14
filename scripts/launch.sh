#!/usr/bin/env bash
# ============================================================================
# Top-level dispatcher — picks a verified production config by name.
#
# Usage:
#   ./scripts/launch.sh bench3-fp8     # bench3 golden (FP8, MBS=10, target 268)
#   ./scripts/launch.sh europe-bf16    # europe baseline (BF16, MBS=8, target 289)
#   ./scripts/launch.sh bench3-smoke   # bench3 7-iter smoke
#   ./scripts/launch.sh gb10           # GB10 single-GPU correctness
#
# See docs/reproducible_runs.md for what each preset means and expected
# output. See docs/production_status.md for throughput references.
# ============================================================================
set -euo pipefail

case "${1:-}" in
  bench3-fp8)   exec "$(dirname "$0")/run_bench3_golden_fp8.sh" ;;
  europe-bf16)  exec "$(dirname "$0")/run_europe_baseline_bf16.sh" ;;
  bench3-smoke) exec "$(dirname "$0")/run_bench3_smoke_quick.sh" ;;
  gb10)         exec "$(dirname "$0")/run_gb10_correctness.sh" ;;
  *)
    echo "usage: $0 {bench3-fp8|europe-bf16|bench3-smoke|gb10}" >&2
    exit 2
    ;;
esac
