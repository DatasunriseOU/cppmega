#!/bin/bash
# Self-logging bench3 golden FP8 smoke launcher.
#
# Every step writes to STEPLOG with timestamp + exit code so failures are
# diagnosable from the log alone. Training tee'd to its own LOG file.
#
# Run from repo root via:
#   nohup bash scripts/run_golden_smoke_logged.sh > /dev/null 2>&1 &
#
# Watch progress:
#   tail -f /home/dave/logs/golden_smoke_steps.log
#   tail -f $(cat /tmp/golden_smoke_log_path)

set -u

STEPLOG=/home/dave/logs/golden_smoke_steps.log
mkdir -p /home/dave/logs

# Tee everything (stdout+stderr) into STEPLOG. Append so re-runs accumulate.
exec > >(tee -a "$STEPLOG") 2>&1

log() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*"; }
step() {
    local name="$1"
    shift
    log "BEGIN $name :: $*"
    "$@"
    local ec=$?
    log "END   $name :: exit=$ec"
    if [ "$ec" -ne 0 ] && [ "$name" != "git_stash" ] && [ "$name" != "git_status" ] && [ "$name" != "launch" ]; then
        log "FATAL step $name failed; aborting."
        exit "$ec"
    fi
    return "$ec"
}

log "===================================================="
log "run_golden_smoke_logged.sh starting"
log "  whoami=$(whoami) pwd=$(pwd) host=$(hostname) date=$(date)"
log "===================================================="

REPO="${REPO:-/mnt/data/cppmega-root/cppmega}"
step cd cd "$REPO"

step git_status git status --short
# Do NOT --include-untracked: dataset cache + checkpoints (~76+ GiB total)
# are untracked on bench3 and would hang git stash for 7+ min per call
# (bench3 incident 2026-04-15). Tracked-only stash is enough — we only
# need to clear modifications to tracked files before pull.
step git_stash git stash push -m "auto-$(date +%s)"
step git_pull git pull origin main
step git_log git log -1 --oneline

TRAIN_LOG="/home/dave/logs/golden_smoke_$(date +%Y%m%d_%H%M%S).log"
echo "$TRAIN_LOG" > /tmp/golden_smoke_log_path
log "TRAIN_LOG=$TRAIN_LOG"

# Golden MBS=10 with selective recompute + MLA fusion + clip-grad + indexer
# loss off (per reference_fp8_mbs10_bench3_wins.md). Earlier MBS=10 OOM
# (commit 93217b0 forensics) was caused by my run_bench3_golden_fp8.sh
# wrapper dropping --recompute-granularity selective from EXTRA_FLAGS;
# fixed in commit 36584ad. Reverting to MBS=10 to match golden 268.
# Start with MBS=8 (safer, gives ~10 GiB margin) since we're now also
# enabling DSA indexer KL loss (coeff=0.001) via lemyx tilelang fused FA+KL
# kernel. Once MBS=8 + indexer loss verified, can escalate to MBS=10.
step launch bash scripts/run_bench3_mbs8_fp8.sh

# On failure, show last 30 lines of the training log to make diagnosis
# possible from STEPLOG alone (no need to ssh in and grep).
TRAIN_INTERNAL_LOG="/mnt/data/cppmega-root/cppmega/cppmega_nam56r_dsa_9_4_fp8_m_v3.log"
log "TRAIN_INTERNAL_LOG=$TRAIN_INTERNAL_LOG (last 30 lines):"
[ -f "$TRAIN_INTERNAL_LOG" ] && tail -30 "$TRAIN_INTERNAL_LOG" | sed 's/^/    /'

log "===================================================="
log "run_golden_smoke_logged.sh DONE"
log "  TRAIN_LOG=$TRAIN_LOG"
log "  STEPLOG=$STEPLOG"
log "===================================================="
