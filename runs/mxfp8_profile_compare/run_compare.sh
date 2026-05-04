#!/usr/bin/env bash
# Run the GB10 NAM56R-quarter shape under three MXFP8 contracts for 20 steps
# each, capture tok/sec, loss-after-20, peak GPU memory.
#
# Usage:  ./runs/mxfp8_profile_compare/run_compare.sh
set -euo pipefail

ROOT="${ROOT:-/home/dave/source/cppmega}"
OUT_DIR="${ROOT}/runs/mxfp8_profile_compare"
mkdir -p "${OUT_DIR}"

# Three contracts ranked by expected memory efficiency: compact_direct_v1
# refuses sidecars and BF16 fallback; gemm_ready_v1 keeps gemm-ready saved
# operands; legacy is the current production default with TE TN-adapter
# sidecars.  All run on the same NAM56R-quarter (13-layer) shape, MTP=2,
# micro_batch=4, seq=4096 → 16384 tokens/step.
declare -a CONFIGS=(
  "compact_direct|--fp8-recipe mxfp8 --mxfp8-linear-kernel-contract compact_direct_v1"
  "gemm_ready_v1|--fp8-recipe mxfp8 --mxfp8-linear-kernel-contract gemm_ready_v1"
  "legacy|--fp8-recipe mxfp8 --mxfp8-linear-kernel-contract legacy"
)

export CPPMEGA_TRAIN_ITERS=20

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

echo
echo "=== summary"
for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%|*}"
  log=$(ls -t "${OUT_DIR}/profile_${name}_"*.log 2>/dev/null | head -1)
  [ -z "${log}" ] && continue
  echo "--- ${name} (${log})"
  grep -E "iteration|loss:|tokens per|consumed samples" "${log}" | tail -5
  nvsmi="${log%.log}.nvsmi.log"
  if [ -f "${nvsmi}" ]; then
    awk -F, '{ if ($2 + 0 > peak) peak = $2 + 0 } END { print "peak_used_mib=" peak }' "${nvsmi}"
  fi
done
