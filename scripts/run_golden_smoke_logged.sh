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
    if [ "$ec" -ne 0 ] && [ "$name" != "git_stash" ] && [ "$name" != "git_status" ]; then
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

step launch bash scripts/launch.sh bench3-smoke

log "===================================================="
log "run_golden_smoke_logged.sh DONE"
log "  TRAIN_LOG=$TRAIN_LOG"
log "  STEPLOG=$STEPLOG"
log "===================================================="
