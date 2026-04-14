#!/usr/bin/env bash
# ============================================================================
# bench3 smoke quick — 7-iter sanity test of the bench3 golden FP8 config
# ============================================================================
#
# Use this to verify a fresh checkout produces sane TFLOP/s before committing
# to a full training run. Identical config to run_bench3_golden_fp8.sh except
# TRAIN_ITERS=7 so it finishes in ~90s (including model load + compile).
#
# Model / hardware / precision: see run_bench3_golden_fp8.sh.
# Target:     TFLOP/s converging toward 260-268 by iter 4-7 (cold iters 1-2
#             dominated by TileLang JIT compile / CUDA graph capture).
# Peak memory: ~115 GiB / 141 GiB per rank.
#
# Interpretation of output:
#   - grad_norm should be finite (< 10), lm_loss ~ 10 (vocab log ~ ln 100k)
#   - "throughput per GPU (TFLOP/s/GPU)" line is the headline metric
#   - peak_alloc_gib < 135 = healthy; > 140 = approaching OOM cliff
#
# If output shows NaN / inf: check that Liger reduction="mean" broadcast
# patch applied (stdout line "MTP Liger CE patch installed"), and that you
# did NOT set CPPMEGA_MTP_NATIVE_HOPPER_CE=1 (produces grad_norm=NaN, see
# docs/production_status.md).
#
# Logs:
#   ${REMOTE_ROOT}/cppmega/cppmega_nam56r_dsa_9_4_fp8_m_v3.log
# ============================================================================
set -euo pipefail

exec env \
  TRAIN_ITERS=7 \
  bash "$(dirname "$0")/run_bench3_golden_fp8.sh"
