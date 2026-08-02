#!/usr/bin/env bash
# Install TileLang 0.1.9 from DatasunriseOU/tilelang@de8bb88c into the active
# (or given) venv.
#
# Primary path: download the prebuilt x86_64 TileLang/tvm-ffi wheel pair.
# Fallback:     clone DatasunriseOU/tilelang at the exact pinned commit.
#
# This fork commit carries apache/tvm#18938 (TVMDerivedObject.__slots__ fix,
# via vendored TVM DatasunriseOU/tvm@e25ca6ae), restores the nvbench CUDA
# L2-cache-flush header, and removes the apache-tvm-ffi<0.1.10 cap (upstream
# PR #2071), so it imports cleanly under tvm-ffi >=0.1.12 as required by FA4
# beta23. Its lazy driver stub also exports cuFuncGetAttribute required by the
# CUDA 13.2 TVM runtime, its matching tvm-ffi wheel is v0.1.13.post5, and its
# CUDA codegen emits scalar TIRx AllocBuffer special scopes correctly.
# Must match STACK.lock.
#
# Usage:
#   scripts/install_tilelang_wheel.sh                # uses $VIRTUAL_ENV
#   scripts/install_tilelang_wheel.sh /path/to/venv  # activates venv first
#
# Env overrides:
#   TILELANG_WHEEL_URL   GS (or https) URL of the prebuilt wheel
#   TVM_FFI_WHEEL_URL    GS (or https) URL of the ABI-matched tvm-ffi wheel
#   TILELANG_GIT_COMMIT  Source-build commit pin (fallback)
#   TILELANG_FORCE_SOURCE=1  Skip wheel, always build from source

set -euo pipefail

WHEEL_URL="${TILELANG_WHEEL_URL:-sftp://BUCKET_ARTIFACTS/tilelang/tilelang-0.1.9-cp38-abi3-linux_x86_64.whl}"
TVM_FFI_WHEEL_URL="${TVM_FFI_WHEEL_URL:-sftp://BUCKET_ARTIFACTS/tilelang/apache_tvm_ffi-0.1.13.post5-cp313-cp313-linux_x86_64.whl}"
GIT_COMMIT="${TILELANG_GIT_COMMIT:-de8bb88cc382b0e78bc804244f79c4be8cc9e75f}"
TVM_COMMIT="${TILELANG_TVM_COMMIT:-e25ca6ae50beee0e907b1e5ed32949879caddde1}"
TVM_FFI_COMMIT="${TILELANG_TVM_FFI_COMMIT:-521efeb30bfd9e4946b248b3d76e6391028233a3}"
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
pip install "z3-solver==4.15.4.0"

# --- helper: verify import succeeds -----------------------------------------
verify_import() {
  python - <<'PY'
from importlib import metadata
import tilelang
import tvm
from tvm.s_tir.analysis import is_pure_function

print(f"[tilelang] imported OK: {tilelang.__version__} @ {tilelang.__file__}")
observed = metadata.version("tilelang")
if observed != "0.1.9":
    raise SystemExit(f"unexpected TileLang version: {observed!r} != '0.1.9'")
observed_ffi = metadata.version("apache-tvm-ffi")
if observed_ffi != "0.1.13.post5":
    raise SystemExit(f"unexpected tvm-ffi version: {observed_ffi!r} != '0.1.13.post5'")
body = tvm.tirx.AttrStmt(
    None, "threadblock_swizzle_pattern", 0, tvm.tirx.Evaluate(0)
)
assert body.node is None
assert is_pure_function(tvm.tirx.PrimFunc([], body))
PY
}

# --- wheel path (x86_64 only) ------------------------------------------------
if [[ "${FORCE_SOURCE}" != "1" && "${ARCH}" == "x86_64" ]]; then
  TMP_WHEEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cppmega-tilelang-wheel.XXXXXX")"
  TMP_WHEEL="${TMP_WHEEL_DIR}/tilelang.whl"
  TMP_FFI_WHEEL="${TMP_WHEEL_DIR}/tvm_ffi.whl"
  cleanup_temp_wheel() {
    find "${TMP_WHEEL_DIR}" -depth -delete
  }
  trap cleanup_temp_wheel EXIT
  echo "[tilelang] fetching wheel: ${WHEEL_URL}"
  if [[ "${WHEEL_URL}" == sftp://* ]]; then
    if gsutil cp "${WHEEL_URL}" "${TMP_WHEEL}" &&
      gsutil cp "${TVM_FFI_WHEEL_URL}" "${TMP_FFI_WHEEL}"; then
      pip install --force-reinstall --no-deps "${TMP_FFI_WHEEL}" "${TMP_WHEEL}"
      pip install "${TMP_WHEEL}"  # resolve deps if any missing
      verify_import
      exit 0
    else
      echo "[tilelang] GS fetch failed, falling back to source build" >&2
    fi
  else
    if curl -fL --retry 3 -o "${TMP_WHEEL}" "${WHEEL_URL}" &&
      curl -fL --retry 3 -o "${TMP_FFI_WHEEL}" "${TVM_FFI_WHEEL_URL}"; then
      pip install --force-reinstall --no-deps "${TMP_FFI_WHEEL}" "${TMP_WHEEL}"
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
test "$(git -C 3rdparty/tvm/3rdparty/tvm-ffi rev-parse HEAD)" = "${TVM_FFI_COMMIT}"
test -f 3rdparty/tvm/3rdparty/nvbench/l2_cache_flush.h
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_APACHE_TVM_FFI=0.1.13.post5 \
  pip install --force-reinstall --no-deps \
    3rdparty/tvm/3rdparty/tvm-ffi
pip install -e .
verify_import
