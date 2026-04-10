#!/usr/bin/env bash
# Sync cppmega code to h200_1.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_1}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_2}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
REMOTE_DIR="${REMOTE_DIR:-${REMOTE_ROOT}/cppmega}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REMOTE_TMP_ARCHIVE="${REMOTE_TMP_ARCHIVE:-/tmp/cppmega-sync.tgz}"
LOCAL_TMP_ARCHIVE="$(mktemp -t cppmega-sync.XXXXXX.tgz)"

trap 'rm -f "${LOCAL_TMP_ARCHIVE}"' EXIT

tar -czf "${LOCAL_TMP_ARCHIVE}" \
  -C "${LOCAL_DIR}" \
  README.md \
  pyproject.toml \
  cppmega \
  docs \
  scripts \
  tests

gcloud compute scp \
  --zone "${REMOTE_ZONE}" \
  "${LOCAL_TMP_ARCHIVE}" \
  "${REMOTE_HOST}:${REMOTE_TMP_ARCHIVE}"

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "set -e; rm -rf '${REMOTE_DIR}'; mkdir -p '${REMOTE_DIR}'; tar -xzf '${REMOTE_TMP_ARCHIVE}' -C '${REMOTE_DIR}'; rm -f '${REMOTE_TMP_ARCHIVE}'; if [ -f '${REMOTE_VENV}/bin/activate' ]; then source '${REMOTE_VENV}/bin/activate' && pip install --no-deps -e '${REMOTE_DIR}' 2>/dev/null; fi"

echo "synced ${LOCAL_DIR} -> ${REMOTE_HOST}:${REMOTE_DIR}"
