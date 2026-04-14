#!/bin/bash
# Clean bisect: start from known-good golden commit 0ce8a3a, verify grad
# finite, then roll forward one commit at a time until NaN appears.
#
# Runs from a SEPARATE worktree at /mnt/data/cppmega-root/cppmega-bisect
# so we don't disturb the main repo at /mnt/data/cppmega-root/cppmega.
#
# Usage on bench3:
#   cd /mnt/data/cppmega-root/cppmega-bisect
#   bash scripts/bisect_nan_golden.sh <commit_sha>
# Self-logs to /home/dave/logs/bisect_nan_<commit>.log.
set -u

COMMIT="${1:-}"
if [ -z "$COMMIT" ]; then
  echo "usage: $0 <commit_sha>" >&2
  exit 2
fi

BISECT_ROOT="${BISECT_ROOT:-/mnt/data/cppmega-root/cppmega-bisect}"
STEPLOG=/home/dave/logs/bisect_nan_${COMMIT}.log
mkdir -p /home/dave/logs
exec > >(tee -a "$STEPLOG") 2>&1

log() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*"; }
step() {
    local name="$1"; shift
    log "BEGIN $name :: $*"
    "$@"
    local ec=$?
    log "END   $name :: exit=$ec"
    return "$ec"
}

log "===== bisect_nan_golden.sh commit=$COMMIT ====="

# Create or reuse worktree
if [ ! -d "$BISECT_ROOT" ]; then
  step worktree_add git -C /mnt/data/cppmega-root/cppmega worktree add -f "$BISECT_ROOT" "$COMMIT"
else
  step worktree_sync bash -c "cd $BISECT_ROOT && git fetch origin main 2>&1 | tail -3 && git checkout -f $COMMIT 2>&1 | tail -3 && git log -1 --oneline"
fi

cd "$BISECT_ROOT"

# Rebuild editable install so Python imports the committed files
step pip_install /mnt/data/venv/bin/pip install --no-deps -e "$BISECT_ROOT" 2>&1 | tail -5

# Also need to force python cache flush — delete __pycache__
step rm_pycache find "$BISECT_ROOT/cppmega" -type d -name __pycache__ -exec rm -rf {} +

# Run 5-iter smoke with full golden env (per reference_fp8_mbs10_bench3_wins.md)
TRAIN_LOG="/home/dave/logs/bisect_train_${COMMIT}.log"
log "TRAIN_LOG=$TRAIN_LOG"

step training env REPO="$BISECT_ROOT" \
  VARIANT=v3 PP_SIZE=1 VPP_SIZE=1 EP_SIZE=8 DP_SIZE=1 TRAIN_ITERS=5 MBS=10 GBS=80 \
  CG_FLAGS=NONE \
  FP8_FLAGS="--fp8-format hybrid --fp8-recipe tensorwise --fp8-amax-history-len 1024 --fp8-amax-compute-algo max" \
  CPPMEGA_MAIN_HEAD_LINEAR_CE=1 CPPMEGA_LINEAR_CE_KERNEL=liger \
  CPPMEGA_MTP_LIGER_CE=1 \
  CPPMEGA_INDEX_CACHE=1 CPPMEGA_LEMYX_DSA=1 \
  CPPMEGA_DSA_INDEXER_LOSS_COEFF=0 CPPMEGA_DSA_SKIP_INDEXER_LOSS=1 \
  EXTRA_FLAGS="--recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj --mla-down-proj-fusion --clip-grad 1.0" \
  bash scripts/remote_smoke_h200_dsa_9_4_m.sh > "$TRAIN_LOG" 2>&1

# Inspect grad_norm
log "==== grad_norm per iter ===="
grep -E "iteration.*throughput|grad norm" "$TRAIN_LOG" | tail -10

# Final verdict
FINITE_COUNT=$(grep -E "iteration.*throughput" "$TRAIN_LOG" | grep -cv "grad norm: nan")
NAN_COUNT=$(grep -E "iteration.*throughput" "$TRAIN_LOG" | grep -c "grad norm: nan")
TOTAL=$(grep -cE "iteration.*throughput" "$TRAIN_LOG")

log "===== VERDICT commit=$COMMIT finite=$FINITE_COUNT nan=$NAN_COUNT total=$TOTAL ====="
if [ "$NAN_COUNT" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
  log "RESULT_GOOD_GRAD_FINITE"
elif [ "$NAN_COUNT" -gt 0 ]; then
  log "RESULT_BAD_GRAD_NAN"
else
  log "RESULT_UNKNOWN_NO_ITERS"
fi
