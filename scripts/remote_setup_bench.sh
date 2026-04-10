#!/usr/bin/env bash
# Setup cppmega environment on h200_1.
#
# This machine has system-wide PyTorch 2.12 + CUDA 12.8 and no pre-existing
# cppmega or Megatron installation.  The script creates a venv from system
# Python, installs Megatron-LM + TE + Author Mamba3, and downloads training
# data from GCS.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_1}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_2}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
REMOTE_CPPMEGA_DIR="${REMOTE_CPPMEGA_DIR:-${REMOTE_ROOT}/cppmega}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_BASE_VENV="${REMOTE_BASE_VENV:-/home/dave/nanochat-exact/venv313}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-bench-setup.sh}"
MEGATRON_COMMIT="${MEGATRON_COMMIT:-e40feed4a}"
MAMBA_COMMIT="${MAMBA_COMMIT:-31f3d7baba69d0ccad1635ace1e477367899e408}"
INSTALL_AUTHOR_MAMBA3="${INSTALL_AUTHOR_MAMBA3:-1}"
INSTALL_TE_PYTORCH="${INSTALL_TE_PYTORCH:-1}"
TE_VERSION="${TE_VERSION:-2.13.0}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
GCS_DATA_BUCKET="${GCS_DATA_BUCKET:-sftp://BUCKET_TRAINING_DATA/data/parquet}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-bench-setup.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'EOF'
set -euo pipefail
mkdir -p "${REMOTE_ROOT}"
cd "${REMOTE_ROOT}"

# ---- Find base Python venv with PyTorch ----
if [ ! -x "${REMOTE_BASE_VENV}/bin/python" ]; then
  echo "ERROR: base venv not found at ${REMOTE_BASE_VENV}" >&2
  exit 1
fi
BASE_PYTHON="${REMOTE_BASE_VENV}/bin/python"
BASE_TORCH_VERSION="$("${BASE_PYTHON}" -c 'import torch; print(torch.__version__)')"
echo "base Python: ${BASE_PYTHON}"
echo "base PyTorch: ${BASE_TORCH_VERSION}"

# ---- Create cppmega venv from base ----
rm -rf "${REMOTE_VENV}"
"${BASE_PYTHON}" -m venv --system-site-packages "${REMOTE_VENV}"
source "${REMOTE_VENV}/bin/activate"
python -m ensurepip --upgrade 2>/dev/null || true
python -m pip install --upgrade pip wheel packaging setuptools 2>/dev/null
python -m pip install --no-deps pybind11

# ---- Link base venv site-packages ----
BASE_SITE="$("${BASE_PYTHON}" -c "import site; [print(p) for p in site.getsitepackages() if 'site-packages' in p]" | head -1)"
VENV_SITE="$(python -c "import site; [print(p) for p in site.getsitepackages() if 'site-packages' in p]" | head -1)"

# Add base venv paths to cppmega venv
PTH_FILE="${VENV_SITE}/cppmega-base-venv.pth"
"${BASE_PYTHON}" -c "
import site, sys
paths = set()
for p in site.getsitepackages():
    paths.add(p)
for p in sys.path:
    if p and '${REMOTE_BASE_VENV}' in p:
        paths.add(p)
print('\n'.join(sorted(paths)))
" > "${PTH_FILE}"
echo "linked base venv paths via ${PTH_FILE}"

# Link NVIDIA packages
if [ -d "${BASE_SITE}/nvidia" ] && [ ! -e "${VENV_SITE}/nvidia" ]; then
  mkdir -p "${VENV_SITE}/nvidia"
  for entry in "${BASE_SITE}/nvidia"/*; do
    [ -e "${entry}" ] || continue
    name="$(basename "${entry}")"
    [ -e "${VENV_SITE}/nvidia/${name}" ] || ln -s "${entry}" "${VENV_SITE}/nvidia/${name}"
  done
  echo "linked NVIDIA packages from base venv"
fi

# ---- Install Megatron-LM ----
if [ ! -d megatron-lm ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git megatron-lm
fi
cd megatron-lm
git fetch --all --tags
git checkout "${MEGATRON_COMMIT}"
python -m pip install --no-deps -e .
cd "${REMOTE_ROOT}"

# ---- Install Transformer Engine ----
if [ "${INSTALL_TE_PYTORCH}" = "1" ]; then
  CUDNN_ROOT=""
  NCCL_ROOT=""
  if [ -d "${BASE_SITE}/nvidia/cudnn" ]; then
    CUDNN_ROOT="${BASE_SITE}/nvidia/cudnn"
  fi
  if [ -d "${BASE_SITE}/nvidia/nccl" ]; then
    NCCL_ROOT="${BASE_SITE}/nvidia/nccl"
  fi
  if [ -n "${CUDNN_ROOT}" ]; then
    export CUDNN_PATH="${CUDNN_ROOT}"
    export CPATH="${CUDNN_ROOT}/include:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDNN_ROOT}/include:${CPLUS_INCLUDE_PATH:-}"
    export LIBRARY_PATH="${CUDNN_ROOT}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  fi
  if [ -n "${NCCL_ROOT}" ]; then
    export NCCL_ROOT_DIR="${NCCL_ROOT}"
    export CPATH="${NCCL_ROOT}/include:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${NCCL_ROOT}/include:${CPLUS_INCLUDE_PATH:-}"
    export LIBRARY_PATH="${NCCL_ROOT}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  fi
  python -m pip install --no-deps --force-reinstall \
    "transformer_engine_cu13==${TE_VERSION}" \
    "transformer_engine_torch==${TE_VERSION}" || echo "WARN: TE cu13/torch install failed; trying pip packages"
  python -m pip install --no-deps \
    "transformer-engine==${TE_VERSION}" onnx onnxscript onnx_ir nvdlfw-inspect ml_dtypes 2>/dev/null || true
  python -c "import transformer_engine; print('TE ready:', transformer_engine.__file__)" || echo "WARN: TE import failed"
fi

# ---- Install Author Mamba3 ----
if [ "${INSTALL_AUTHOR_MAMBA3}" = "1" ]; then
  cd "${REMOTE_ROOT}"
  if [ ! -d state-spaces-mamba ]; then
    git clone https://github.com/state-spaces/mamba.git state-spaces-mamba
  fi
  cd state-spaces-mamba
  git fetch --all --tags
  git checkout "${MAMBA_COMMIT}"
  MAMBA_FORCE_BUILD=TRUE python -m pip install --no-deps --no-build-isolation --force-reinstall --ignore-installed .
  python -c "
import importlib.util
assert importlib.util.find_spec('mamba_ssm.modules.mamba3') is not None, \
    'Author Mamba3 install did not expose mamba_ssm.modules.mamba3'
print('Author Mamba3 ready')
"
fi

# ---- Install flash-attn ----
python -m pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARN: flash-attn install failed"

# ---- Install TileLang for Mamba3 MIMO kernels (requires apache-tvm-ffi<0.1.10) ----
# tvm-ffi 0.1.10 introduced __slots__=() enforcement via _ObjectSlotsMeta that breaks
# TileLang's TVMDerivedObject. TileLang's vendored TVM fork doesn't have the
# apache/tvm PR #18938 fix. Pin tvm-ffi<0.1.10 (what tilelang main branch does).
python -m pip install --no-deps "apache-tvm-ffi<0.1.10" 2>/dev/null || \
  echo "WARN: apache-tvm-ffi pin failed"
python -m pip install --no-deps tilelang 2>/dev/null || \
  echo "WARN: tilelang install failed"
python -c "
import tilelang, importlib.metadata
tvm_ffi_ver = importlib.metadata.version('apache-tvm-ffi')
assert tvm_ffi_ver < '0.1.10', f'tvm-ffi must be <0.1.10, got {tvm_ffi_ver}'
print(f'TileLang {tilelang.__version__} + tvm-ffi {tvm_ffi_ver} ready')
" || echo "WARN: tilelang verification failed"

# ---- Install cppmega ----
cd "${REMOTE_ROOT}"
mkdir -p "${REMOTE_CPPMEGA_DIR}"
if [ -f "${REMOTE_CPPMEGA_DIR}/pyproject.toml" ]; then
  python -m pip install --no-deps -e "${REMOTE_CPPMEGA_DIR}"
else
  echo "WARN: cppmega not synced yet; run remote_sync_bench.sh first"
fi

# ---- Download training data from GCS ----
if [ "${DOWNLOAD_DATA}" = "1" ]; then
  DATA_DIR="${REMOTE_ROOT}/data/parquet"
  mkdir -p "${DATA_DIR}"

  echo "downloading clang_semantic_4k_v10..."
  gcloud storage cp -r "${GCS_DATA_BUCKET}/clang_semantic_4k_v10/" "${DATA_DIR}/clang_semantic_4k_v10/" 2>/dev/null || \
    echo "WARN: clang_semantic_4k_v10 download failed"

  echo "downloading clang_commits_4k_v1..."
  gcloud storage cp -r "${GCS_DATA_BUCKET}/clang_commits_4k_v1/" "${DATA_DIR}/clang_commits_4k_v1/" 2>/dev/null || \
    echo "WARN: clang_commits_4k_v1 download failed"

  echo "data download complete: $(du -sh "${DATA_DIR}" 2>/dev/null | cut -f1)"
fi

# ---- Create triton cache dir ----
mkdir -p "${REMOTE_ROOT}/.triton-cache"

# ---- Verification ----
CLONED_TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
if [ "${CLONED_TORCH_VERSION}" != "${BASE_TORCH_VERSION}" ]; then
  echo "WARNING: venv torch drifted: ${CLONED_TORCH_VERSION} != ${BASE_TORCH_VERSION}" >&2
fi

python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA devices: {torch.cuda.device_count()}')
try:
    import megatron.core
    print(f'megatron-core: OK')
except ImportError:
    print('megatron-core: NOT FOUND')
try:
    import transformer_engine
    print(f'transformer_engine: OK')
except ImportError:
    print('transformer_engine: NOT FOUND (optional)')
try:
    import mamba_ssm.modules.mamba3
    print('mamba3: OK')
except ImportError:
    print('mamba3: NOT FOUND (optional)')
try:
    import cppmega
    print('cppmega: OK')
except ImportError:
    print('cppmega: NOT synced yet')
"
echo "bench setup complete"
EOF

gcloud compute scp \
  --zone "${REMOTE_ZONE}" \
  "${LOCAL_TMP_SCRIPT}" \
  "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_CPPMEGA_DIR='${REMOTE_CPPMEGA_DIR}' REMOTE_BASE_VENV='${REMOTE_BASE_VENV}' REMOTE_VENV='${REMOTE_VENV}' MEGATRON_COMMIT='${MEGATRON_COMMIT}' MAMBA_COMMIT='${MAMBA_COMMIT}' INSTALL_AUTHOR_MAMBA3='${INSTALL_AUTHOR_MAMBA3}' INSTALL_TE_PYTORCH='${INSTALL_TE_PYTORCH}' TE_VERSION='${TE_VERSION}' DOWNLOAD_DATA='${DOWNLOAD_DATA}' GCS_DATA_BUCKET='${GCS_DATA_BUCKET}' bash '${REMOTE_TMP_SCRIPT}'" \
  || { status=$?; echo "setup failed (exit ${status})"; exit $status; }

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "rm -f '${REMOTE_TMP_SCRIPT}'"

echo "bench setup done on ${REMOTE_HOST}"
