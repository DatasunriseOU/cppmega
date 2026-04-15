#!/bin/bash
# Bisect NaN grad_norm across commits. Runs against a SEPARATE worktree at
# /mnt/data/cppmega-root/cppmega-bisect so the main repo at
# /mnt/data/cppmega-root/cppmega is untouched.
#
# Usage on bench3:
#   cd /mnt/data/cppmega-root/cppmega
#   bash scripts/bisect_nan_golden.sh <commit_sha>
# Self-logs to /home/dave/logs/bisect_nan_<commit>.log.
#
# Prior version had three bugs:
#   1) PYTHONPATH shadowed the worktree by prepending the MAIN repo; imports
#      resolved to main-repo code regardless of commit under test.
#   2) VARIANT=v3 was introduced at commit f6d6bb1; older commits' launchers
#      reject it. We now overlay main's launcher into the worktree at test
#      time so v3 + bench3 autodetect always work.
#   3) Pre-flight did not verify python was actually importing the worktree.
# This version fixes all three and aborts with a clear error if the pre-check
# fails.

set -u

COMMIT="${1:-}"
if [ -z "$COMMIT" ]; then
  echo "usage: $0 <commit_sha>" >&2
  exit 2
fi

MAIN_REPO="${MAIN_REPO:-/mnt/data/cppmega-root/cppmega}"
BISECT_ROOT="${BISECT_ROOT:-/mnt/data/cppmega-root/cppmega-bisect}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data/cppmega-root}"
REMOTE_VENV="${REMOTE_VENV:-/mnt/data/venv}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${REMOTE_ROOT}/megatron-lm}"

# Pin the launcher script from MAIN tree (has v3 variant + bench3 autodetect).
LAUNCHER_SRC="${LAUNCHER_SRC:-${MAIN_REPO}/scripts/remote_smoke_h200_dsa_9_4_m.sh}"

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
log "MAIN_REPO=${MAIN_REPO}"
log "BISECT_ROOT=${BISECT_ROOT}"
log "REMOTE_ROOT=${REMOTE_ROOT}"
log "REMOTE_VENV=${REMOTE_VENV}"
log "LAUNCHER_SRC=${LAUNCHER_SRC}"

# ---------------------------------------------------------------------------
# 1) Create or reuse worktree, check out target commit
# ---------------------------------------------------------------------------
if [ ! -d "$BISECT_ROOT" ]; then
  step worktree_add git -C "$MAIN_REPO" worktree add -f "$BISECT_ROOT" "$COMMIT"
else
  step worktree_sync bash -c "cd $BISECT_ROOT && git fetch origin main 2>&1 | tail -3 && git checkout -f $COMMIT 2>&1 | tail -3 && git log -1 --oneline"
fi

cd "$BISECT_ROOT"

# ---------------------------------------------------------------------------
# 2) Overlay main's launcher onto worktree so v3 + bench3 autodetect work
#    even on commits that predate those features.
# ---------------------------------------------------------------------------
if [ ! -f "$LAUNCHER_SRC" ]; then
  log "FATAL: LAUNCHER_SRC not found at $LAUNCHER_SRC"
  exit 3
fi
step copy_launcher cp "$LAUNCHER_SRC" "$BISECT_ROOT/scripts/remote_smoke_h200_dsa_9_4_m.sh"
chmod +x "$BISECT_ROOT/scripts/remote_smoke_h200_dsa_9_4_m.sh"

# ---------------------------------------------------------------------------
# 3) Rebuild editable install pointing at BISECT_ROOT, then clear pycache.
#    --force-reinstall makes sure the .pth and egg-info rewrite the worktree
#    path (previous runs may have left them pointing at main repo).
# ---------------------------------------------------------------------------
step pip_install_bisect "${REMOTE_VENV}/bin/pip" install --no-deps --force-reinstall -e "$BISECT_ROOT" 2>&1 | tail -8
step rm_pycache_cppmega bash -c "find \"$BISECT_ROOT/cppmega\" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true"
# Also nuke main-repo pycache so there's no stale compiled bytecode lurking.
step rm_pycache_main bash -c "find \"$MAIN_REPO/cppmega\" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true"

# ---------------------------------------------------------------------------
# 4) PRE-CHECK: verify python imports cppmega from the BISECT worktree.
#    PYTHONPATH must put worktree FIRST; filesystem path imports take
#    precedence over .pth-installed finders.
# ---------------------------------------------------------------------------
export PYTHONPATH="${BISECT_ROOT}:${MEGATRON_ROOT}"
log "PYTHONPATH=${PYTHONPATH}"

log "BEGIN precheck_import_cppmega"
PRECHECK_OUT=$("${REMOTE_VENV}/bin/python" -c "
import cppmega, os, sys
path = cppmega.__file__
print('cppmega.__file__=' + path)
print('BISECT_ROOT=' + os.environ.get('BISECT_ROOT',''))
if 'cppmega-bisect' not in path:
    print('IMPORT_WRONG: expected cppmega-bisect in path, got ' + path, file=sys.stderr)
    sys.exit(42)
print('IMPORT_OK')
" 2>&1)
PRECHECK_EC=$?
echo "$PRECHECK_OUT"
log "END   precheck_import_cppmega :: exit=${PRECHECK_EC}"
if [ "$PRECHECK_EC" -ne 0 ]; then
  log "FATAL: cppmega import did not resolve to bisect worktree. Aborting."
  log "RESULT_ABORT_IMPORT_WRONG"
  exit 4
fi

# Same check for megatron.core (if applicable)
"${REMOTE_VENV}/bin/python" -c "
import megatron.core as mc
print('megatron.core.__file__=' + mc.__file__)
" 2>&1 | tee -a "$STEPLOG" || true

# ---------------------------------------------------------------------------
# 5) Run 5-iter smoke with golden env (FP8 tensorwise MBS=10 EP=8).
#    Pass PYTHONPATH through explicitly so the launcher does not clobber it.
# ---------------------------------------------------------------------------
TRAIN_LOG="/home/dave/logs/bisect_train_${COMMIT}.log"
log "TRAIN_LOG=$TRAIN_LOG"

step training env \
  REPO="$BISECT_ROOT" \
  REMOTE_ROOT="$REMOTE_ROOT" \
  REMOTE_VENV="$REMOTE_VENV" \
  BISECT_PYTHONPATH="${BISECT_ROOT}:${MEGATRON_ROOT}" \
  VARIANT=v3 PP_SIZE=1 VPP_SIZE=1 EP_SIZE=8 DP_SIZE=1 TRAIN_ITERS=5 MBS=10 GBS=80 \
  CG_FLAGS=NONE \
  FP8_FLAGS="--fp8-format hybrid --fp8-recipe tensorwise --fp8-amax-history-len 1024 --fp8-amax-compute-algo max" \
  CPPMEGA_MAIN_HEAD_LINEAR_CE=1 CPPMEGA_LINEAR_CE_KERNEL=liger \
  CPPMEGA_MTP_LIGER_CE=1 \
  CPPMEGA_INDEX_CACHE=1 CPPMEGA_LEMYX_DSA=1 \
  CPPMEGA_DSA_INDEXER_LOSS_COEFF=0 CPPMEGA_DSA_SKIP_INDEXER_LOSS=1 \
  EXTRA_FLAGS="--recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj --mla-down-proj-fusion --clip-grad 1.0" \
  bash -c 'export PYTHONPATH="${BISECT_PYTHONPATH}:${PYTHONPATH:-}"; bash "'"$BISECT_ROOT"'/scripts/remote_smoke_h200_dsa_9_4_m.sh"' \
  > "$TRAIN_LOG" 2>&1

# ---------------------------------------------------------------------------
# 6) Post-check: log where training actually imported cppmega from (from the
#    TRAIN_LOG's first torchrun banner or by a separate one-liner).
# ---------------------------------------------------------------------------
log "==== post-run import check ===="
"${REMOTE_VENV}/bin/python" -c "import cppmega; print('post:', cppmega.__file__)" 2>&1 | tail -3

log "==== grad_norm per iter ===="
grep -E "iteration.*throughput|grad norm" "$TRAIN_LOG" | tail -10

# ---------------------------------------------------------------------------
# 7) Final verdict
# ---------------------------------------------------------------------------
FINITE_COUNT=$(grep -E "iteration.*throughput" "$TRAIN_LOG" | grep -cv "grad norm: nan")
NAN_COUNT=$(grep -E "iteration.*throughput" "$TRAIN_LOG" | grep -c "grad norm: nan")
TOTAL=$(grep -cE "iteration.*throughput" "$TRAIN_LOG")
TFLOP_SAMPLE=$(grep -oE "TFLOP/s per GPU[^|]*" "$TRAIN_LOG" | tail -3 | tr '\n' ' ')

log "===== VERDICT commit=$COMMIT finite=$FINITE_COUNT nan=$NAN_COUNT total=$TOTAL TFLOPs=${TFLOP_SAMPLE} ====="
if [ "$NAN_COUNT" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
  log "RESULT_GOOD_GRAD_FINITE"
elif [ "$NAN_COUNT" -gt 0 ]; then
  log "RESULT_BAD_GRAD_NAN"
else
  log "RESULT_UNKNOWN_NO_ITERS"
fi
