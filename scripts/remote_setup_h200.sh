#!/usr/bin/env bash
set -euo pipefail

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
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-setup.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'EOF'
set -euo pipefail
mkdir -p "${REMOTE_ROOT}"
cd "${REMOTE_ROOT}"

if [ ! -d "${REMOTE_BASE_VENV}" ]; then
  echo "missing base nanochat venv at ${REMOTE_BASE_VENV}" >&2
  exit 1
fi

BASE_TORCH_VERSION="$("${REMOTE_BASE_VENV}/bin/python" -c 'import torch; print(torch.__version__)')"

rm -rf "${REMOTE_VENV}"
mkdir -p "${REMOTE_VENV}"
cp -a "${REMOTE_BASE_VENV}/." "${REMOTE_VENV}/"
find "${REMOTE_VENV}" -path '*/site-packages/~vidia*' -prune -exec rm -rf {} + || true

OLD_VENV="${REMOTE_BASE_VENV}" NEW_VENV="${REMOTE_VENV}" "${REMOTE_BASE_VENV}/bin/python" - <<'PY'
from pathlib import Path
import os

old = os.environ["OLD_VENV"]
new = os.environ["NEW_VENV"]

targets = [Path(new) / "pyvenv.cfg", *(Path(new) / "bin").iterdir()]
for path in targets:
    if not path.is_file():
        continue
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        continue
    if old in text:
        path.write_text(text.replace(old, new))
PY

CLONED_PREFIX="$("${REMOTE_VENV}/bin/python" -c 'import sys; print(sys.prefix)')"
if [ "${CLONED_PREFIX}" != "${REMOTE_VENV}" ]; then
  echo "cloned venv prefix mismatch: ${CLONED_PREFIX} != ${REMOTE_VENV}" >&2
  exit 1
fi

source "${REMOTE_VENV}/bin/activate"

python -m pip install --no-deps pybind11

if ! command -v python3-config >/dev/null 2>&1; then
  cat > "${REMOTE_VENV}/bin/python3-config" <<'PYCFG'
#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--extension-suffix" ]; then
  python3 - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or "")
PY
  exit 0
fi

echo "unsupported python3-config invocation: $*" >&2
exit 1
PYCFG
  chmod +x "${REMOTE_VENV}/bin/python3-config"
fi

if [ ! -d megatron-lm ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git megatron-lm
fi
cd megatron-lm
git fetch --all --tags
git checkout "${MEGATRON_COMMIT}"
python -m pip install --upgrade pip wheel packaging
python -m pip install --no-deps -e .

if [ "${INSTALL_AUTHOR_MAMBA3}" = "1" ]; then
  cd "${REMOTE_ROOT}"
  if [ ! -d state-spaces-mamba ]; then
    git clone https://github.com/state-spaces/mamba.git state-spaces-mamba
  fi
  cd state-spaces-mamba
  git fetch --all --tags
  git checkout "${MAMBA_COMMIT}"
  MAMBA_FORCE_BUILD=TRUE python -m pip install --no-deps --no-build-isolation .
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
  --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_CPPMEGA_DIR='${REMOTE_CPPMEGA_DIR}' REMOTE_BASE_VENV='${REMOTE_BASE_VENV}' REMOTE_VENV='${REMOTE_VENV}' MEGATRON_COMMIT='${MEGATRON_COMMIT}' MAMBA_COMMIT='${MAMBA_COMMIT}' INSTALL_AUTHOR_MAMBA3='${INSTALL_AUTHOR_MAMBA3}' bash '${REMOTE_TMP_SCRIPT}'"

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "rm -f '${REMOTE_TMP_SCRIPT}'"
