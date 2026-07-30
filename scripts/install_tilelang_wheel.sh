#!/usr/bin/env bash
# Install TileLang 0.1.9 from DatasunriseOU/tilelang@b2545eaa into the active
# (or given) venv.
#
# Primary path: download prebuilt x86_64 wheel from GS, pip install.
# Fallback:     clone DatasunriseOU/tilelang at the exact pinned commit.
#
# This fork commit carries apache/tvm#18938 (TVMDerivedObject.__slots__ fix,
# via vendored TVM DatasunriseOU/tvm@78f930ed), restores the nvbench CUDA
# L2-cache-flush header, and removes the apache-tvm-ffi<0.1.10 cap (upstream
# PR #2071), so it imports cleanly under tvm-ffi >=0.1.12 as required by FA4
# beta23. Its lazy driver stub also exports cuFuncGetAttribute required by the
# CUDA 13.2 TVM runtime. Must match STACK.lock.
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

WHEEL_URL="${TILELANG_WHEEL_URL:-sftp://BUCKET_ARTIFACTS/tilelang/tilelang-0.1.9-cp38-abi3-linux_x86_64.whl}"
GIT_COMMIT="${TILELANG_GIT_COMMIT:-b2545eaa3f11610a31e5b8371aab97c369e95f27}"
TVM_COMMIT="${TILELANG_TVM_COMMIT:-78f930edc805920428388518e12d111019383d2f}"
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
from importlib import metadata
import tilelang

print(f"[tilelang] imported OK: {tilelang.__version__} @ {tilelang.__file__}")
observed = metadata.version("tilelang")
if observed != "0.1.9":
    raise SystemExit(f"unexpected TileLang version: {observed!r} != '0.1.9'")
PY
}

# --- wheel path (x86_64 only) ------------------------------------------------
if [[ "${FORCE_SOURCE}" != "1" && "${ARCH}" == "x86_64" ]]; then
  TMP_WHEEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cppmega-tilelang-wheel.XXXXXX")"
  TMP_WHEEL="${TMP_WHEEL_DIR}/tilelang.whl"
  cleanup_temp_wheel() {
    rm -f "${TMP_WHEEL}"
    rmdir "${TMP_WHEEL_DIR}" 2>/dev/null || true
  }
  trap cleanup_temp_wheel EXIT
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
  git clone --recursive https://github.com/DatasunriseOU/tilelang.git "${SRC_DIR}"
fi
cd "${SRC_DIR}"
git fetch origin
git checkout "${GIT_COMMIT}"
test "$(git rev-parse HEAD)" = "${GIT_COMMIT}"
git submodule update --init --recursive
test "$(git -C 3rdparty/tvm rev-parse HEAD)" = "${TVM_COMMIT}"
test -f 3rdparty/tvm/3rdparty/nvbench/l2_cache_flush.h
pip install -e . --no-build-isolation
verify_import
