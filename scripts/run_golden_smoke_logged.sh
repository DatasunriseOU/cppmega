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
step git_stash git stash push -m "auto-$(date +%s)" --include-untracked
step git_pull git pull origin main
step git_log git log -1 --oneline

TRAIN_LOG="/home/dave/logs/golden_smoke_$(date +%Y%m%d_%H%M%S).log"
echo "$TRAIN_LOG" > /tmp/golden_smoke_log_path
log "TRAIN_LOG=$TRAIN_LOG"

# Switched bench3-smoke (MBS=10) → bench3-mbs8 because golden MBS=10 OOMs
# at 134.65 GiB / 139.80 GiB cap on bench3 (verified 2026-04-15). MBS=8
# gives ~10 GiB margin. See scripts/run_bench3_mbs8_fp8.sh header.
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
