#!/usr/bin/env bash
# Robust detached launch wrapper for remote smoke scripts.
#
# Addresses two classes of bugs:
#
#   1. SSH session teardown killing tmux/nohup children.  `setsid` creates
#      a new session; combined with `disown` and closed stdin/stderr/stdout,
#      the child survives SSH disconnect.
#
#   2. Accidental foreground timeouts killing training mid-warmup.  Our
#      smoke scripts need 3+ minutes of JIT warmup before the first
#      training iter prints — agents that use `timeout 30` never see
#      training output and conclude "script hung".  This wrapper launches
#      in the background immediately and leaves a PID file for monitoring.
#
# Usage:
#   bash scripts/remote_launch_detached.sh <log-file> <env-assignments...> -- <script-path> [script-args...]
#
# Example:
#   bash scripts/remote_launch_detached.sh /tmp/p1_full_baseline.log \
#     CPPMEGA_INDEX_CACHE=1 CPPMEGA_LEMYX_DSA=1 TRAIN_ITERS=30 \
#     -- scripts/remote_smoke_h200_dsa_9_4_m.sh
#
# Creates:
#   <log-file>                 — combined stdout+stderr
#   <log-file>.pid             — background PID (empty if launch failed)
#   <log-file>.status          — "RUNNING" during run, "DONE <exit>" after
#   <log-file>.started         — timestamp of launch
#
# Monitor with:
#   tail -f <log-file>
#   cat <log-file>.status
#   grep -q '^DONE' <log-file>.status  # true when complete
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <log-file> <env=val> ... -- <script-path> [script-args...]" >&2
  exit 2
fi

LOG_FILE="$1"
shift

# Collect env assignments until we see --
ENV_ASSIGNS=()
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--" ]; then
    shift
    break
  fi
  ENV_ASSIGNS+=("$1")
  shift
done

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <log-file> <env=val> ... -- <script-path> [script-args...]" >&2
  exit 2
fi

SCRIPT_AND_ARGS=("$@")

PID_FILE="${LOG_FILE}.pid"
STATUS_FILE="${LOG_FILE}.status"
STARTED_FILE="${LOG_FILE}.started"

# Clear any previous runs' state files so a monitor can distinguish.
rm -f "${PID_FILE}" "${STATUS_FILE}"
: > "${LOG_FILE}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STARTED_FILE}"
echo "RUNNING" > "${STATUS_FILE}"

# Build the inner command.  Use `env` so each KEY=VAL is applied to the
# target script only (not to this wrapper).
INNER_CMD=(env "${ENV_ASSIGNS[@]}" bash "${SCRIPT_AND_ARGS[@]}")

# setsid detaches from the controlling terminal / SSH session.
# Redirect all std{in,out,err} so SIGHUP on SSH-close doesn't propagate.
# `disown` makes the shell forget the job.
(
  setsid "${INNER_CMD[@]}" >>"${LOG_FILE}" 2>&1 </dev/null &
  CHILD_PID=$!
  echo "${CHILD_PID}" > "${PID_FILE}"
  disown "${CHILD_PID}" 2>/dev/null || true

  # Watchdog: poll the child PID and update status file when it exits.
  # `wait` can't be used because the watchdog runs in a sibling subshell
  # where ${CHILD_PID} is not its own child, so `wait` returns immediately.
  (
    while kill -0 "${CHILD_PID}" 2>/dev/null; do sleep 5; done
    # Can't read exit code cross-process; report as DONE unknown.
    echo "DONE" > "${STATUS_FILE}"
  ) >/dev/null 2>&1 &
  disown $! 2>/dev/null || true
) </dev/null >/dev/null 2>&1

# Give setsid a moment to actually start.
sleep 1

PID=$(cat "${PID_FILE}" 2>/dev/null || echo "")
if [ -z "${PID}" ] || ! kill -0 "${PID}" 2>/dev/null; then
  echo "LAUNCH FAILED: no PID or process died before check" >&2
  echo "DONE -1" > "${STATUS_FILE}"
  exit 1
fi

echo "LAUNCHED pid=${PID}  log=${LOG_FILE}"
echo "Monitor: tail -f ${LOG_FILE}  |  status: cat ${STATUS_FILE}"
