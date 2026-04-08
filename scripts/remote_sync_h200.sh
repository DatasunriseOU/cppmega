#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_legacy}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_3}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/data/cppmega}"
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

read -r -d '' REMOTE_CMD <<'EOF' || true
set -euo pipefail
rm -rf "${REMOTE_DIR}"
mkdir -p "${REMOTE_DIR}"
tar -xzf "${REMOTE_TMP_ARCHIVE}" -C "${REMOTE_DIR}"
rm -f "${REMOTE_TMP_ARCHIVE}"
EOF

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "REMOTE_DIR='${REMOTE_DIR}' REMOTE_TMP_ARCHIVE='${REMOTE_TMP_ARCHIVE}' bash -lc $(printf '%q' "${REMOTE_CMD}")"

echo "synced ${LOCAL_DIR} -> ${REMOTE_HOST}:${REMOTE_DIR}"
