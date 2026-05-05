#!/usr/bin/env bash
# Batch=16 follow-up: bf16 vs mxfp8_gemm_ready under higher compute density.
# Each config gets its own process so JIT/cuBLAS state doesn't leak across runs
# (we hit a cuBLAS internal error in the back-to-back batch=4 sweep).
set -euo pipefail

ROOT="${ROOT:-/home/dave/source/cppmega}"
OUT_DIR="${ROOT}/runs/mxfp8_profile_compare"
mkdir -p "${OUT_DIR}"

declare -a CONFIGS=(
  "bf16_b16|--train-iters 20 --micro-batch-size 16 --global-batch-size 16 --fp8-recipe off"
  "mxfp8_gemm_ready_b16|--train-iters 20 --micro-batch-size 16 --global-batch-size 16 --fp8-recipe mxfp8 --mxfp8-linear-kernel-contract gemm_ready_v1"
)

for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%|*}"
  args="${cfg#*|}"
  ts=$(date +%Y%m%d_%H%M%S)
  RUN_ID="profile_${name}_${ts}"
  LOG="${OUT_DIR}/${RUN_ID}.log"
  NVSMI_LOG="${OUT_DIR}/${RUN_ID}.nvsmi.log"
  echo "=== running ${name} → ${LOG}"
  RUN_ID="${RUN_ID}" LOG="${LOG}" NVSMI_LOG="${NVSMI_LOG}" \
    "${ROOT}/scripts/local_gb10_quarter_train.sh" ${args} \
    || echo "  WARN: ${name} exited non-zero (kept log)"
  echo "  done ${name}"
done
