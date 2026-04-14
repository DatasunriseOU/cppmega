#!/usr/bin/env bash
# ============================================================================
# LEGACY / DEPRECATED (2026-04-14)
# ----------------------------------------------------------------------------
# Targets the original bring-up host `h200_legacy` (LOCATION_3)
# via gcloud scp + ssh. Current active H200 anchors (2026-04) are bench3
# (LOCATION_1) and europe (LOCATION_2); modern workflow uses direct
# `dave@<ip>` ssh with the in-place tmux "remote body" scripts (no separate
# sync step needed). Override REMOTE_HOST / REMOTE_ZONE / REMOTE_DIR
# explicitly to target bench3/europe if you still want the gcloud-based
# sync path. The LOCATION_3 defaults below are preserved only for
# backward compatibility with any caller that still exports them.
#
# GB10 note: `ssh gb10` may fail with hostname resolution (e.g.
# `ssh dave@gx10-9cd4: hostname resolution`). GB10 requires local-network
# access or a direct IP; this sync script does not target GB10.
# ============================================================================
set -euo pipefail

# LEGACY default (LOCATION_3); override REMOTE_HOST/REMOTE_ZONE for bench3 or europe.
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
