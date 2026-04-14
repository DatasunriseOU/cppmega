#!/usr/bin/env bash
# Bench3 H200x8 test runner: SparseMLA FP8 + TileLang SparseMLA verification.
#
# Two tests:
#   Test 1: SparseMLA_FP8 always active (CPPMEGA_DSA_FP8_ATTENTION=1)
#   Test 2: TileLang SparseMLA (BF16) + compile + IndexCache + Liger CE
#
# Both tests use: PP=2 VPP=2 EP=4 h32 MBS=4, 10 iterations.
#
# Run on bench3 (h200_1):
#   bash scripts/bench3_fp8_attn_and_tilelang_tests.sh
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data/cppmega-root}"
REMOTE_VENV="${REMOTE_VENV:-/mnt/data/venv}"

echo "=== Bench3 FP8 Attention + TileLang SparseMLA Tests ==="
echo "REMOTE_ROOT=${REMOTE_ROOT}"
echo "REMOTE_VENV=${REMOTE_VENV}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Sync code
# ---------------------------------------------------------------------------
echo "=== Step 1: git pull ==="
cd "${REMOTE_ROOT}/cppmega"
git pull
echo ""

# ---------------------------------------------------------------------------
# Step 2: Apply all upstream patches
# ---------------------------------------------------------------------------
echo "=== Step 2: Apply upstream patches ==="
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"

# venv NVIDIA libs
_NV="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia"
_LD_PREFIX=""
for _pkg in cudnn nccl cublas; do
  _d="${_NV}/${_pkg}/lib"
  [ -d "${_d}" ] && _LD_PREFIX="${_d}:${_LD_PREFIX}"
done
export LD_LIBRARY_PATH="${_LD_PREFIX}/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"

python -m cppmega.megatron.upstream_patches.apply_dsa_cg_patches
echo ""

# ---------------------------------------------------------------------------
# Step 3: Check and fix mamba_ssm mamba3.py DT fp32 + GQA backward
# ---------------------------------------------------------------------------
echo "=== Step 3: Check mamba_ssm fixes ==="

MAMBA3_PY="${REMOTE_VENV}/lib/python3.13/site-packages/mamba_ssm/modules/mamba3.py"
MIMO_BWD_PY="${REMOTE_VENV}/lib/python3.13/site-packages/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"

# Check DT fp32: softplus(dd_dt + dt_bias) should cast to float32 before softplus
echo "--- Checking DT fp32 in mamba3.py ---"
if grep -q 'softplus.*dd_dt.*dt_bias' "${MAMBA3_PY}" 2>/dev/null; then
  if grep 'softplus.*dd_dt.*dt_bias' "${MAMBA3_PY}" | grep -q 'float32\|\.float()'; then
    echo "  OK: DT fp32 fix already applied"
  else
    echo "  FIXING: Adding .to(torch.float32) before softplus for DT"
    # Apply the fix: F.softplus(dd_dt + self.dt_bias) -> F.softplus((dd_dt + self.dt_bias).to(torch.float32))
    sed -i.bak 's/F\.softplus(dd_dt + self\.dt_bias)/F.softplus((dd_dt + self.dt_bias).to(torch.float32))/g' "${MAMBA3_PY}"
    echo "  Verifying fix:"
    grep 'softplus.*dd_dt.*dt_bias' "${MAMBA3_PY}" | head -2
  fi
else
  echo "  SKIP: softplus(dd_dt + dt_bias) pattern not found in mamba3.py"
fi

# Check GQA backward fix
echo "--- Checking GQA backward in mamba3_mimo_bwd.py ---"
if [ -f "${MIMO_BWD_PY}" ]; then
  if grep -q 'G < H.*dq.*view.*sum' "${MIMO_BWD_PY}" 2>/dev/null || \
     grep -q 'elif G < H' "${MIMO_BWD_PY}" 2>/dev/null; then
    echo "  OK: GQA backward fix already applied"
  else
    echo "  WARNING: GQA backward fix may not be applied."
    echo "  Checking current G/H handling:"
    grep -n 'G.*H\|ngroups.*nheads\|dq.*view\|ValueError.*G value' "${MIMO_BWD_PY}" | head -10
    echo ""
    echo "  If G<H is not supported, the fix is:"
    echo "    elif G < H: dq = dq_tilelang.view(B, S, R, G, H//G, N).sum(dim=4)"
    echo "  Applied manually on europe; bench3 memory notes say it should already be patched."
  fi
else
  echo "  SKIP: ${MIMO_BWD_PY} not found"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Clear caches
# ---------------------------------------------------------------------------
echo "=== Step 4: Clear pycache + tilelang cache ==="
find "${REMOTE_ROOT}/cppmega" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${REMOTE_ROOT}/megatron-lm" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf "${REMOTE_ROOT}/.triton-cache" 2>/dev/null || true
rm -rf /tmp/.tilelang_cache 2>/dev/null || true
rm -rf /tmp/tilelang_* 2>/dev/null || true
echo "  Caches cleared."
echo ""

# ---------------------------------------------------------------------------
# Test 1: SparseMLA_FP8 always active
# ---------------------------------------------------------------------------
echo "================================================================"
echo "=== TEST 1: SparseMLA_FP8 (CPPMEGA_DSA_FP8_ATTENTION=1) ==="
echo "================================================================"
echo ""

# Clear caches before test 1
find "${REMOTE_ROOT}/cppmega" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf "${REMOTE_ROOT}/.triton-cache" 2>/dev/null || true
rm -rf /tmp/.tilelang_cache /tmp/tilelang_* 2>/dev/null || true

TEST1_EXIT=0
REMOTE_ROOT="${REMOTE_ROOT}" \
REMOTE_VENV="${REMOTE_VENV}" \
VARIANT=v1 \
TRAIN_ITERS=10 \
MBS=4 \
CPPMEGA_DSA_FP8_ATTENTION=1 \
CPPMEGA_INDEX_CACHE=1 \
CPPMEGA_DSA_INDEXER_LOSS_COEFF=0 \
RUN_ID="bench3_test1_fp8_attn" \
LOG="${REMOTE_ROOT}/cppmega/bench3_test1_fp8_attn.log" \
  bash "${REMOTE_ROOT}/cppmega/scripts/remote_smoke_h200_dsa_9_4_m.sh" || TEST1_EXIT=$?

echo ""
echo "=== Test 1 exit code: ${TEST1_EXIT} ==="
echo "Log: ${REMOTE_ROOT}/cppmega/bench3_test1_fp8_attn.log"
echo ""

# ---------------------------------------------------------------------------
# Test 2: TileLang SparseMLA (BF16) + compile + IndexCache + Liger CE
# ---------------------------------------------------------------------------
echo "================================================================"
echo "=== TEST 2: TileLang SparseMLA + compile + IndexCache + Liger CE ==="
echo "================================================================"
echo ""

# Clear caches before test 2
find "${REMOTE_ROOT}/cppmega" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf "${REMOTE_ROOT}/.triton-cache" 2>/dev/null || true
rm -rf /tmp/.tilelang_cache /tmp/tilelang_* 2>/dev/null || true

TEST2_EXIT=0
REMOTE_ROOT="${REMOTE_ROOT}" \
REMOTE_VENV="${REMOTE_VENV}" \
VARIANT=v1 \
TRAIN_ITERS=10 \
MBS=4 \
CPPMEGA_DSA_FP8_ATTENTION=0 \
CPPMEGA_DSA_SPARSE_MODE=tilelang \
CPPMEGA_INDEX_CACHE=1 \
CPPMEGA_DSA_INDEXER_LOSS_COEFF=0 \
RUN_ID="bench3_test2_tilelang_compile" \
LOG="${REMOTE_ROOT}/cppmega/bench3_test2_tilelang_compile.log" \
  bash "${REMOTE_ROOT}/cppmega/scripts/remote_smoke_h200_dsa_9_4_m.sh" || TEST2_EXIT=$?

echo ""
echo "=== Test 2 exit code: ${TEST2_EXIT} ==="
echo "Log: ${REMOTE_ROOT}/cppmega/bench3_test2_tilelang_compile.log"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "================================================================"
echo "=== SUMMARY ==="
echo "================================================================"
echo ""
echo "Test 1 (SparseMLA_FP8):"
echo "  Log: ${REMOTE_ROOT}/cppmega/bench3_test1_fp8_attn.log"
if [ -f "${REMOTE_ROOT}/cppmega/bench3_test1_fp8_attn.log" ]; then
  echo "  --- Throughput lines ---"
  grep -i "throughput\|tok/sec\|TFLOP\|tflops" "${REMOTE_ROOT}/cppmega/bench3_test1_fp8_attn.log" | tail -5
  echo "  --- SparseMLA activation ---"
  grep -i "SparseMLA\|FP8.*applied\|FP8.*active\|sparse_mla" "${REMOTE_ROOT}/cppmega/bench3_test1_fp8_attn.log" | head -5
fi
echo ""
echo "Test 2 (TileLang SparseMLA + compile):"
echo "  Log: ${REMOTE_ROOT}/cppmega/bench3_test2_tilelang_compile.log"
if [ -f "${REMOTE_ROOT}/cppmega/bench3_test2_tilelang_compile.log" ]; then
  echo "  --- Throughput lines ---"
  grep -i "throughput\|tok/sec\|TFLOP\|tflops" "${REMOTE_ROOT}/cppmega/bench3_test2_tilelang_compile.log" | tail -5
  echo "  --- SparseMLA + compile activation ---"
  grep -i "SparseMLA\|fused_sparse_mla_absorbed\|compile.*applied\|compile.*patch\|mamba3_compile" "${REMOTE_ROOT}/cppmega/bench3_test2_tilelang_compile.log" | head -5
fi
echo ""
echo "Test 1 exit: ${TEST1_EXIT}, Test 2 exit: ${TEST2_EXIT}"
if [ "${TEST1_EXIT}" -ne 0 ] || [ "${TEST2_EXIT}" -ne 0 ]; then
  echo "WARNING: One or more tests failed!"
  exit 1
fi
echo "=== All tests passed ==="
