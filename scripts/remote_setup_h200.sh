#!/usr/bin/env bash
# ============================================================================
# LEGACY / DEPRECATED (2026-04-14)
# ----------------------------------------------------------------------------
# This script was written for the original bring-up host `h200_legacy`
# (LOCATION_3) in early 2026 and targets `/mnt/data` on that machine.
#
# Current active H200 anchors (2026-04) are:
#   bench3  — zone LOCATION_1, hostname h200_1,
#             root /mnt/data/cppmega-root, venv /mnt/data/venv
#   europe  — zone LOCATION_2, hostname h200_1,
#             root /home/dave/cppmega-root, venv /home/dave/cppmega-root/cppmega-venv
#
# The LOCATION_3 defaults below are preserved for backward compatibility with
# any caller still exporting them explicitly; new work should override
# REMOTE_HOST / REMOTE_ZONE / REMOTE_ROOT to point at bench3 or europe, or
# (for bench3/europe) use the in-place tmux "remote body" scripts such as
# scripts/remote_smoke_h200_dsa_9_4_m.sh instead of this gcloud-wrapping setup.
# ============================================================================
set -euo pipefail

# LEGACY default (LOCATION_3); override REMOTE_HOST/REMOTE_ZONE for bench3 or europe.
REMOTE_HOST="${REMOTE_HOST:-h200_legacy}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_3}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
REMOTE_CPPMEGA_DIR="${REMOTE_CPPMEGA_DIR:-${REMOTE_ROOT}/cppmega}"
REMOTE_BASE_VENV="${REMOTE_BASE_VENV:-${REMOTE_ROOT}/venv}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-setup.sh}"
MEGATRON_COMMIT="${MEGATRON_COMMIT:-fd762549816ea21cc8c00db602a66c01717e2794}"
MAMBA_COMMIT="${MAMBA_COMMIT:-31f3d7baba69d0ccad1635ace1e477367899e408}"
INSTALL_AUTHOR_MAMBA3="${INSTALL_AUTHOR_MAMBA3:-0}"
INSTALL_TE_PYTORCH="${INSTALL_TE_PYTORCH:-0}"
TE_VERSION="${TE_VERSION:-2.13.0}"
GCS_ARTIFACT_PREFIX="${GCS_ARTIFACT_PREFIX:-}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-setup.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'EOF'
set -euo pipefail
mkdir -p "${REMOTE_ROOT}"
cd "${REMOTE_ROOT}"

if [ ! -x "${REMOTE_BASE_VENV}/bin/python" ]; then
  echo "missing base python at ${REMOTE_BASE_VENV}/bin/python" >&2
  exit 1
fi

BASE_TORCH_VERSION="$(${REMOTE_BASE_VENV}/bin/python -c 'import torch; print(torch.__version__)')"

rm -rf "${REMOTE_VENV}"
"${REMOTE_BASE_VENV}/bin/python" -m venv --system-site-packages "${REMOTE_VENV}"

source "${REMOTE_VENV}/bin/activate"
python -m ensurepip --upgrade
python -m pip install --upgrade pip wheel packaging setuptools
python -m pip install --no-deps pybind11

BASE_SITE_PACKAGES="$(${REMOTE_BASE_VENV}/bin/python - <<'PY'
import site
paths = site.getsitepackages()
print("\n".join(paths))
PY
)"
BASE_PY_SITE="$(${REMOTE_BASE_VENV}/bin/python - <<'PY'
import site
for path in site.getsitepackages():
    if 'site-packages' in path:
        print(path)
        break
PY
)"
REMOTE_SITE_PACKAGES="$(python - <<'PY'
import site
for path in site.getsitepackages():
    if 'site-packages' in path:
        print(path)
        break
PY
)"
PTH_FILE="${REMOTE_SITE_PACKAGES}/cppmega-base-venv.pth"
printf '%s\n' "${BASE_SITE_PACKAGES}" > "${PTH_FILE}"
"${REMOTE_BASE_VENV}/bin/python" - "${PTH_FILE}" <<'PY'
import site
import sys

base_paths = set(site.getsitepackages())
target = sys.argv[1]
with open(target, "a", encoding="utf-8") as handle:
    for path in sys.path:
        if path and path.startswith("/mnt/data/venv/") and path not in base_paths:
            handle.write(path + "\n")
PY

if [ -d "${BASE_PY_SITE}/nvidia" ]; then
  mkdir -p "${REMOTE_SITE_PACKAGES}/nvidia"
  for entry in "${BASE_PY_SITE}/nvidia"/*; do
    [ -e "${entry}" ] || continue
    name="$(basename "${entry}")"
    if [ ! -e "${REMOTE_SITE_PACKAGES}/nvidia/${name}" ]; then
      ln -s "${entry}" "${REMOTE_SITE_PACKAGES}/nvidia/${name}"
    fi
  done
fi

cat > "${REMOTE_VENV}/bin/python3-config" <<'PYCFG'
#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--extension-suffix" ]; then
  python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or "")
PY
  exit 0
fi

echo "unsupported python3-config invocation: $*" >&2
exit 1
PYCFG
chmod +x "${REMOTE_VENV}/bin/python3-config"

if [ ! -d megatron-lm ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git megatron-lm
fi
cd megatron-lm
git fetch --all --tags
git checkout "${MEGATRON_COMMIT}"
python -m pip install --no-deps -e .

if [ "${INSTALL_TE_PYTORCH}" = "1" ]; then
  CUDNN_ROOT="${BASE_PY_SITE}/nvidia/cudnn"
  NCCL_ROOT="${BASE_PY_SITE}/nvidia/nccl"
  export CUDNN_PATH="${CUDNN_ROOT}"
  export NCCL_ROOT_DIR="${NCCL_ROOT}"
  export CPATH="${CUDNN_ROOT}/include:${NCCL_ROOT}/include:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="${CUDNN_ROOT}/include:${NCCL_ROOT}/include:${CPLUS_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${CUDNN_ROOT}/lib:${NCCL_ROOT}/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  python -m pip install --no-deps --force-reinstall "transformer_engine_cu13==${TE_VERSION}" "transformer_engine_torch==${TE_VERSION}"
  python -m pip install --no-deps "transformer-engine==${TE_VERSION}" onnx onnxscript onnx_ir nvdlfw-inspect ml_dtypes
  python - <<'PY'
import torch
import transformer_engine
import transformer_engine.pytorch
print('transformer_engine ready', torch.__version__, transformer_engine.__file__)
PY
  if [ -n "${GCS_ARTIFACT_PREFIX}" ]; then
    ART_DIR="${REMOTE_ROOT}/cppmega/te-artifacts"
    mkdir -p "${ART_DIR}"
    python -m pip wheel --no-deps --wheel-dir "${ART_DIR}" "transformer_engine_cu13==${TE_VERSION}" "transformer_engine_torch==${TE_VERSION}" "transformer-engine==${TE_VERSION}" onnx onnxscript onnx_ir nvdlfw-inspect ml_dtypes
    gcloud storage cp "${ART_DIR}"/* "${GCS_ARTIFACT_PREFIX}/"
  fi
fi

if [ "${INSTALL_AUTHOR_MAMBA3}" = "1" ]; then
  cd "${REMOTE_ROOT}"
  if [ ! -d state-spaces-mamba ]; then
    git clone https://github.com/state-spaces/mamba.git state-spaces-mamba
  fi
  cd state-spaces-mamba
  git fetch --all --tags
  git checkout "${MAMBA_COMMIT}"
  MAMBA_FORCE_BUILD=TRUE python -m pip install --no-deps --no-build-isolation --force-reinstall --ignore-installed .
  python - <<'PY'
import importlib.util

assert importlib.util.find_spec("mamba_ssm.modules.mamba3") is not None, (
    "author Mamba3 install did not expose mamba_ssm.modules.mamba3"
)
PY
fi

mkdir -p "${REMOTE_CPPMEGA_DIR}"
if [ -f "${REMOTE_CPPMEGA_DIR}/pyproject.toml" ]; then
  python -m pip install --no-deps -e "${REMOTE_CPPMEGA_DIR}"
else
  echo "warning: missing cppmega checkout at ${REMOTE_CPPMEGA_DIR}; run scripts/remote_sync_h200.sh first" >&2
fi

CLONED_TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
if [ "${CLONED_TORCH_VERSION}" != "${BASE_TORCH_VERSION}" ]; then
  echo "cloned venv torch drifted: ${CLONED_TORCH_VERSION} != ${BASE_TORCH_VERSION}" >&2
  exit 1
fi

python -c "import cppmega, megatron, torch; print('remote setup complete', torch.__version__)"
EOF

gcloud compute scp \
  --zone "${REMOTE_ZONE}" \
  "${LOCAL_TMP_SCRIPT}" \
  "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_CPPMEGA_DIR='${REMOTE_CPPMEGA_DIR}' REMOTE_BASE_VENV='${REMOTE_BASE_VENV}' REMOTE_VENV='${REMOTE_VENV}' MEGATRON_COMMIT='${MEGATRON_COMMIT}' MAMBA_COMMIT='${MAMBA_COMMIT}' INSTALL_AUTHOR_MAMBA3='${INSTALL_AUTHOR_MAMBA3}' INSTALL_TE_PYTORCH='${INSTALL_TE_PYTORCH}' TE_VERSION='${TE_VERSION}' GCS_ARTIFACT_PREFIX='${GCS_ARTIFACT_PREFIX}' bash '${REMOTE_TMP_SCRIPT}'"

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "rm -f '${REMOTE_TMP_SCRIPT}'"
