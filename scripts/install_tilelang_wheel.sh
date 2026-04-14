#!/usr/bin/env bash
# Install TileLang 0.1.8+cuda.gitf309d814 into the active (or given) venv.
#
# Primary path: download prebuilt x86_64 wheel from GS, pip install.
# Fallback:     clone tile-ai/tilelang at commit f309d814, build from source.
#
# Usage:
#   scripts/install_tilelang_wheel.sh                # uses $VIRTUAL_ENV
#   scripts/install_tilelang_wheel.sh /path/to/venv  # activates venv first
#
# Env overrides:
#   TILELANG_WHEEL_URL   GS (or https) URL of the prebuilt wheel
#   TILELANG_GIT_COMMIT  Source-build commit pin (fallback)
#   TILELANG_FORCE_SOURCE=1  Skip wheel, always build from source

set -euo pipefail

WHEEL_URL="${TILELANG_WHEEL_URL:-sftp://BUCKET_ARTIFACTS/tilelang/tilelang-0.1.8+cuda.gitf309d814-cp38-abi3-linux_x86_64.whl}"
GIT_COMMIT="${TILELANG_GIT_COMMIT:-f309d814}"
FORCE_SOURCE="${TILELANG_FORCE_SOURCE:-0}"

# --- venv activation ---------------------------------------------------------
if [[ $# -ge 1 ]]; then
  VENV="$1"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: no active venv. Activate one or pass venv path as \$1." >&2
  exit 1
fi

echo "[tilelang] target venv: ${VIRTUAL_ENV}"
echo "[tilelang] python:      $(python --version 2>&1)"
echo "[tilelang] arch:        $(uname -m)"

ARCH="$(uname -m)"

# --- helper: verify import succeeds -----------------------------------------
verify_import() {
  python - <<'PY'
import tilelang, sys
print(f"[tilelang] imported OK: {tilelang.__version__} @ {tilelang.__file__}")
if "f309d814" not in tilelang.__version__:
    print(f"WARNING: version does not contain expected commit f309d814", file=sys.stderr)
PY
}

# --- wheel path (x86_64 only) ------------------------------------------------
if [[ "${FORCE_SOURCE}" != "1" && "${ARCH}" == "x86_64" ]]; then
  TMP_WHEEL="/tmp/tilelang-${GIT_COMMIT}.whl"
  echo "[tilelang] fetching wheel: ${WHEEL_URL}"
  if [[ "${WHEEL_URL}" == sftp://* ]]; then
    if gsutil cp "${WHEEL_URL}" "${TMP_WHEEL}"; then
      pip install --force-reinstall --no-deps "${TMP_WHEEL}"
      pip install "${TMP_WHEEL}"  # resolve deps if any missing
      verify_import
      exit 0
    else
      echo "[tilelang] GS fetch failed, falling back to source build" >&2
    fi
  else
    if curl -fL --retry 3 -o "${TMP_WHEEL}" "${WHEEL_URL}"; then
      pip install --force-reinstall --no-deps "${TMP_WHEEL}"
      pip install "${TMP_WHEEL}"
      verify_import
      exit 0
    else
      echo "[tilelang] HTTPS fetch failed, falling back to source build" >&2
    fi
  fi
fi

# --- source build fallback ---------------------------------------------------
echo "[tilelang] building from source at commit ${GIT_COMMIT}"
SRC_DIR="${TILELANG_SRC_DIR:-${HOME}/tilelang-build}"
if [[ ! -d "${SRC_DIR}/.git" ]]; then
  git clone --recursive https://github.com/tile-ai/tilelang.git "${SRC_DIR}"
fi
cd "${SRC_DIR}"
git fetch origin
git checkout "${GIT_COMMIT}"
git submodule update --init --recursive
pip install -e . --no-build-isolation
verify_import
