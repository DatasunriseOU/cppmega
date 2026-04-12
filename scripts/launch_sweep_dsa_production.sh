#!/usr/bin/env bash
# Local orchestrator: sync cppmega to bench3, then launch the DSA production
# parameter sweep on the remote machine.
#
# Usage (from local machine):
#   bash scripts/launch_sweep_dsa_production.sh
#   # Resume from a specific config:
#   START_FROM=B4 bash scripts/launch_sweep_dsa_production.sh
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_1}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_1}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
REMOTE_CPPMEGA="${REMOTE_ROOT}/cppmega-root/cppmega"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/venv}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
START_FROM="${START_FROM:-B1}"

echo "[local] Syncing cppmega to ${REMOTE_HOST}..."

# Step 1: git pull on bench3 (the user asked for "git pull to latest commit first")
echo "[local] Running git pull on bench3..."
gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "cd '${REMOTE_CPPMEGA}' && git pull --ff-only origin main 2>&1 | tail -5 && echo 'git pull ok: $(git rev-parse --short HEAD)'"

# Step 2: Sync local uncommitted changes (scripts, cppmega module, tests)
LOCAL_TMP_ARCHIVE="$(mktemp -t cppmega-sweep-sync.XXXXXX.tgz)"
trap 'rm -f "${LOCAL_TMP_ARCHIVE}"' EXIT

tar -czf "${LOCAL_TMP_ARCHIVE}" \
  -C "${LOCAL_DIR}" \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='.omx' \
  --exclude='.tmp' \
  --exclude='*.egg-info' \
  pyproject.toml \
  cppmega \
  scripts \
  tests

REMOTE_TMP_ARCHIVE="/tmp/cppmega-sweep-sync.tgz"
gcloud compute scp \
  --zone "${REMOTE_ZONE}" \
  "${LOCAL_TMP_ARCHIVE}" \
  "${REMOTE_HOST}:${REMOTE_TMP_ARCHIVE}"

# Extract over existing checkout (preserves .git, data, etc.)
gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "tar -xzf '${REMOTE_TMP_ARCHIVE}' -C '${REMOTE_CPPMEGA}' && rm -f '${REMOTE_TMP_ARCHIVE}' && echo 'sync done: $(ls -la ${REMOTE_CPPMEGA}/scripts/remote_sweep_h200_dsa_production.sh 2>/dev/null && echo ok || echo missing)'"

# Step 3: Re-install cppmega in editable mode
gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "source '${REMOTE_VENV}/bin/activate' && pip install --no-deps -e '${REMOTE_CPPMEGA}' 2>&1 | tail -3"

# Step 4: Verify the sweep script exists
echo "[local] Verifying sweep script on bench3..."
gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "test -f '${REMOTE_CPPMEGA}/scripts/remote_sweep_h200_dsa_production.sh' && echo 'sweep script present' || (echo 'ERROR: sweep script missing'; exit 1)"

# Step 5: Launch the driver in a tmux session on bench3
echo "[local] Launching sweep driver on bench3 in tmux session 'dsa_sweep_driver'..."
gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "
    tmux kill-session -t dsa_sweep_driver 2>/dev/null || true
    tmux new-session -d -s dsa_sweep_driver \
      'START_FROM=${START_FROM} bash -l ${REMOTE_CPPMEGA}/scripts/drive_sweep_dsa_production.sh 2>&1 | tee ${REMOTE_ROOT}/cppmega-root/cppmega/nam56r_dsa_sweep/driver.log; echo DRIVER_DONE_\$?'
    echo 'tmux session dsa_sweep_driver launched'
    tmux ls
  "

echo ""
echo "[local] Sweep launched on ${REMOTE_HOST}."
echo "[local] Monitor progress:"
echo "  gcloud compute ssh ${REMOTE_HOST} --zone ${REMOTE_ZONE} --command 'tmux attach -t dsa_sweep_driver'"
echo "  gcloud compute ssh ${REMOTE_HOST} --zone ${REMOTE_ZONE} --command 'cat ${REMOTE_ROOT}/cppmega-root/cppmega/nam56r_dsa_sweep/sweep_results.csv'"
echo "  gcloud compute ssh ${REMOTE_HOST} --zone ${REMOTE_ZONE} --command 'tail -50 ${REMOTE_ROOT}/cppmega-root/cppmega/nam56r_dsa_sweep/nam56r_dsa_sweep_B1.log'"
echo ""
echo "[local] When complete, fetch results:"
echo "  gcloud compute ssh ${REMOTE_HOST} --zone ${REMOTE_ZONE} --command 'cat ${REMOTE_ROOT}/cppmega-root/cppmega/nam56r_dsa_sweep/sweep_results.csv'"
