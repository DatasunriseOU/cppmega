#!/usr/bin/env bash
# Rebuild the TileLang wheel for the FA4 beta23 stack.
#
# Why this exists:
#   FA4 beta23 requires apache-tvm-ffi>=0.1.12. TileLang 0.1.8@f309d814
#   (upstream tile-ai/tilelang) crashes on import under tvm-ffi >=0.1.10:
#       AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'
#   Root cause: tvm-ffi 0.1.10+ enforces __slots__=() on Object subclasses
#   (_ObjectSlotsMeta), but TileLang's vendored TVM (882a774) lacks the fix
#   from apache/tvm#18938 (TVMDerivedObject.__slots__ = ("__dict__","__weakref__")).
#   Upstream tile-ai/tilelang HEAD still caps apache-tvm-ffi<0.1.12, so it is
#   NOT usable. The DatasunriseOU/tilelang fork at fff5cfcc is the clean path:
#     - carries upstream tile-ai/tilelang#2071 (removes the <0.1.10 cap)
#     - vendored TVM submodule = DatasunriseOU/tvm@ada9cffb, which includes
#       apache/tvm#18938 (the __slots__ fix) and restores the nvbench CUDA
#       L2-cache-flush header that TVM still compiles
#     - vendored tvm-ffi = DatasunriseOU/tvm-ffi@4e74cb45
#       (v0.1.13.post1), exactly matching the linked TVM runtime ABI
#     - adapts AdjustMatmulOrder to the fork's boxed-Integer permute_dims API
#     - completes the lazy CUDA driver stub for CUDA 13.2 TVM
#   This script clones that fork commit, ensures the TVM submodule is checked
#   out at the fixed commit, builds the wheel, and drops it in wheels/.
#
# Output: matching apache_tvm_ffi and tilelang wheels under wheels/.
#
# Must match STACK.lock wheels.tilelang (repo/ref/version) and the
# .github/workflows/build-wheels.yml tilelang matrix entry.
#
# Usage:
#   scripts/rebuild_tilelang_wheel.sh
#
# Env overrides:
#   TILELANG_REPO    fork repo URL   (default: DatasunriseOU/tilelang)
#   TILELANG_REF     fork commit     (default: fff5cfcc...)
#   TILELANG_TVM_REF vendored TVM commit (default: ada9cffb...)
#   TILELANG_TVM_FFI_REF vendored tvm-ffi commit (default: 4e74cb45...)
#   TILELANG_SRC_DIR clone dir        (default: $HOME/tilelang-build)
#   WHEELS_DIR       output dir       (default: <repo>/wheels)

set -euo pipefail

TILELANG_REPO="${TILELANG_REPO:-https://github.com/DatasunriseOU/tilelang.git}"
TILELANG_REF="${TILELANG_REF:-fff5cfcc60fed16d163f13cca991256b6ebe1573}"
# DatasunriseOU/tvm commit that includes apache/tvm#18938 (44dbd138d) and
# restores nvbench/l2_cache_flush.h. This is the exact submodule pin recorded
# in the fork's 3rdparty/tvm gitlink at TILELANG_REF.
TILELANG_TVM_REF="${TILELANG_TVM_REF:-ada9cffbb381695651e265039f77c326c146d6b7}"
TILELANG_TVM_FFI_REF="${TILELANG_TVM_FFI_REF:-4e74cb45fbcf6117b69a9864bbe5548f1a7e17a2}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELS_DIR="${WHEELS_DIR:-${REPO_ROOT}/wheels}"
SRC_DIR="${TILELANG_SRC_DIR:-${HOME}/tilelang-build}"

echo "[rebuild-tilelang] repo:      ${TILELANG_REPO}"
echo "[rebuild-tilelang] ref:       ${TILELANG_REF}"
echo "[rebuild-tilelang] tvm sub:   ${TILELANG_TVM_REF}"
echo "[rebuild-tilelang] ffi sub:   ${TILELANG_TVM_FFI_REF}"
echo "[rebuild-tilelang] src dir:   ${SRC_DIR}"
echo "[rebuild-tilelang] wheels:    ${WHEELS_DIR}"

# --- clone / update source ---------------------------------------------------
if [[ ! -d "${SRC_DIR}/.git" ]]; then
  echo "[rebuild-tilelang] cloning ${TILELANG_REPO}"
  git clone "${TILELANG_REPO}" "${SRC_DIR}"
fi
cd "${SRC_DIR}"
git fetch origin
git checkout "${TILELANG_REF}"

# --- ensure vendored TVM carries the apache/tvm#18938 __slots__ fix ----------
# The fork's .gitmodules points 3rdparty/tvm at DatasunriseOU/tvm. Init the
# submodule, then hard-pin it to TILELANG_TVM_REF so the build cannot silently
# use a TVM that lacks the fix.
git submodule update --init --recursive 3rdparty/tvm
cd 3rdparty/tvm
git fetch origin
git checkout "${TILELANG_TVM_REF}"
cd "${SRC_DIR}"
test "$(git -C 3rdparty/tvm/3rdparty/tvm-ffi rev-parse HEAD)" = "${TILELANG_TVM_FFI_REF}"

# Verify the __slots__ fix is actually present in the checked-out TVM. Without
# this, the wheel imports under tvm-ffi >=0.1.10 and crashes at runtime.
if ! grep -q '__slots__ = tuple(slots)' 3rdparty/tvm/python/tvm/runtime/support.py; then
  echo "[rebuild-tilelang] ERROR: vendored TVM at ${TILELANG_TVM_REF} is missing" >&2
  echo "  the apache/tvm#18938 __slots__ fix in python/tvm/runtime/support.py." >&2
  echo "  Refusing to build a wheel that will crash under tvm-ffi >=0.1.10." >&2
  exit 1
fi
echo "[rebuild-tilelang] vendored TVM carries apache/tvm#18938 (__slots__) fix: OK"

# TileLang's TVM runtime still includes this exact nvbench header. Fail before
# the expensive wheel build if the submodule regresses to the broken pin.
if [[ ! -f 3rdparty/tvm/3rdparty/nvbench/l2_cache_flush.h ]]; then
  echo "[rebuild-tilelang] ERROR: vendored TVM at ${TILELANG_TVM_REF} is missing" >&2
  echo "  3rdparty/nvbench/l2_cache_flush.h required by the CUDA runtime." >&2
  exit 1
fi
echo "[rebuild-tilelang] vendored TVM carries the nvbench CUDA header: OK"

# Verify the apache-tvm-ffi cap is gone (upstream PR #2071). If the fork still
# pins <0.1.10, pip will refuse to co-install with FA4 beta23's >=0.1.12.
if grep -q 'apache-tvm-ffi.*<0.1.10' pyproject.toml; then
  echo "[rebuild-tilelang] ERROR: pyproject.toml still caps apache-tvm-ffi<0.1.10." >&2
  echo "  This wheel would be uninstallable alongside FA4 beta23 (needs >=0.1.12)." >&2
  exit 1
fi
echo "[rebuild-tilelang] apache-tvm-ffi cap removed: OK"

# --- build deps (mirror build-wheels.yml) ------------------------------------
python -m pip install --upgrade pip setuptools wheel
python -m pip install --retries 10 --timeout 60 \
  ninja packaging pybind11 cmake numpy einops \
  scikit-build-core setuptools-scm "z3-solver>=4.13.0,<4.15.5" "cython>=3.1.0"

# --- build the ABI-matched wheels -------------------------------------------
mkdir -p "${WHEELS_DIR}"
echo "[rebuild-tilelang] building matching tvm-ffi wheel → ${WHEELS_DIR}"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_APACHE_TVM_FFI=0.1.13.post1 \
  pip wheel 3rdparty/tvm/3rdparty/tvm-ffi \
    --no-build-isolation --no-deps -w "${WHEELS_DIR}" -v
echo "[rebuild-tilelang] building TileLang wheel → ${WHEELS_DIR}"
pip wheel . --no-build-isolation --no-deps -w "${WHEELS_DIR}" -v

# --- report ------------------------------------------------------------------
echo "[rebuild-tilelang] done. Produced:"
ls -lh "${WHEELS_DIR}"/apache_tvm_ffi-*.whl
ls -lh "${WHEELS_DIR}"/tilelang-*.whl
