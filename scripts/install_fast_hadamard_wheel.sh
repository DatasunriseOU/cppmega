#!/usr/bin/env bash
# Install fast_hadamard_transform 1.1.0 (cp313 / linux_x86_64) into the active
# (or given) venv.
#
# PyPI ships only cp310/cp311 (and nothing for cp313). The prebuilt wheel in
# sftp://BUCKET_ARTIFACTS/wheels/ was produced locally on bench3 against
# cu13.2 / torch 2.12 nightly / Python 3.13.
#
# Primary path: download prebuilt wheel from GS (or HTTPS mirror), pip install.
# Fallback:     clone Dao-AILab/fast-hadamard-transform, pip install -e .
#
# Usage:
#   scripts/install_fast_hadamard_wheel.sh                # uses $VIRTUAL_ENV
#   scripts/install_fast_hadamard_wheel.sh /path/to/venv  # activates venv first
#
# Env overrides:
#   FHT_WHEEL_URL        sftp:// or https:// URL of the prebuilt wheel
#   FHT_GIT_COMMIT       Source-build commit pin (fallback; default: HEAD)
#   FHT_FORCE_SOURCE=1   Skip wheel, always build from source

set -euo pipefail

WHEEL_URL="${FHT_WHEEL_URL:-sftp://BUCKET_ARTIFACTS/wheels/fast_hadamard_transform-1.1.0-cp313-cp313-linux_x86_64.whl}"
HTTPS_MIRROR="https://storage.googleapis.com/BUCKET_ARTIFACTS/wheels/fast_hadamard_transform-1.1.0-cp313-cp313-linux_x86_64.whl"
GIT_COMMIT="${FHT_GIT_COMMIT:-}"
FORCE_SOURCE="${FHT_FORCE_SOURCE:-0}"

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

echo "[fast_hadamard_transform] target venv: ${VIRTUAL_ENV}"
echo "[fast_hadamard_transform] python:      $(python --version 2>&1)"
echo "[fast_hadamard_transform] arch:        $(uname -m)"

ARCH="$(uname -m)"
PYTAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"

verify_import() {
  python - <<'PY'
import fast_hadamard_transform, sys
print(f"[fast_hadamard_transform] imported OK: {fast_hadamard_transform.__version__ if hasattr(fast_hadamard_transform, '__version__') else '(no __version__)'} @ {fast_hadamard_transform.__file__}")
PY
}

# --- wheel path (x86_64 / cp313 only) ----------------------------------------
if [[ "${FORCE_SOURCE}" != "1" && "${ARCH}" == "x86_64" && "${PYTAG}" == "cp313" ]]; then
  TMP_WHEEL="/tmp/fast_hadamard_transform-1.1.0-${PYTAG}.whl"
  echo "[fast_hadamard_transform] fetching wheel: ${WHEEL_URL}"
  FETCHED=0
  if [[ "${WHEEL_URL}" == sftp://* ]]; then
    if gsutil cp "${WHEEL_URL}" "${TMP_WHEEL}"; then
      FETCHED=1
    else
      echo "[fast_hadamard_transform] GS fetch failed, trying HTTPS mirror" >&2
      if curl -fL --retry 3 -o "${TMP_WHEEL}" "${HTTPS_MIRROR}"; then
        FETCHED=1
      fi
    fi
  else
    if curl -fL --retry 3 -o "${TMP_WHEEL}" "${WHEEL_URL}"; then
      FETCHED=1
    fi
  fi
  if [[ "${FETCHED}" == "1" ]]; then
    pip install --force-reinstall --no-deps "${TMP_WHEEL}"
    verify_import
    exit 0
  fi
  echo "[fast_hadamard_transform] wheel fetch failed, falling back to source build" >&2
elif [[ "${PYTAG}" != "cp313" ]]; then
  echo "[fast_hadamard_transform] python tag ${PYTAG} != cp313; wheel not applicable, building from source"
fi

# --- source build fallback ---------------------------------------------------
echo "[fast_hadamard_transform] building from source${GIT_COMMIT:+ at commit ${GIT_COMMIT}}"
SRC_DIR="${FHT_SRC_DIR:-${HOME}/fast-hadamard-transform-build}"
if [[ ! -d "${SRC_DIR}/.git" ]]; then
  git clone https://github.com/Dao-AILab/fast-hadamard-transform.git "${SRC_DIR}"
fi
cd "${SRC_DIR}"
git fetch origin
if [[ -n "${GIT_COMMIT}" ]]; then
  git checkout "${GIT_COMMIT}"
fi
pip install -e . --no-build-isolation
verify_import
