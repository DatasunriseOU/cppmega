#!/usr/bin/env bash
set -euo pipefail

CPPMEGA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CPPMEGA_SCRIPT_DIR}/lib/deprecated_guard.sh"
cppmega_deprecated_script_guard "$(basename "$0")" \
  "current bench3/europe in-place smoke scripts"

REMOTE_HOST="${REMOTE_HOST:-h200_legacy}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_3}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_structure_ingress_smoke}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-structure-ingress-smoke.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-structure-ingress.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"
mkdir -p "${REMOTE_ROOT}/cppmega" "${REMOTE_ROOT}/.triton-cache"
REMOTE_WORKDIR="$(mktemp -d /tmp/cppmega-structure-ingress.XXXXXX)"
trap 'rm -rf "${REMOTE_WORKDIR}"' EXIT

python -c "import cppmega, megatron; print('import smoke ok', cppmega.__version__)"
cp "${REMOTE_ROOT}/megatron-lm/pretrain_mamba.py" "${REMOTE_WORKDIR}/pretrain_mamba.py"

cat > "${REMOTE_WORKDIR}/mamba_builders.py" <<'PY'
from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder
PY

python - <<'PY'
from pathlib import Path
p = Path('/mnt/data/megatron-lm/model_provider.py')
text = p.read_text()
old = 'from megatron.training import get_args\n'
new = old + 'from cppmega.megatron.structure_batch import maybe_set_structure_inputs\n'
if 'from cppmega.megatron.structure_batch import maybe_set_structure_inputs\n' not in text:
    text = text.replace(old, new, 1)
needle = '    return model_builder(\n        args,\n        pre_process,\n        post_process,\n        vp_stage,\n        config=config,\n        pg_collection=pg_collection,\n    )\n'
replacement = (
    '    model = model_builder(\n'
    '        args,\n'
    '        pre_process,\n'
    '        post_process,\n'
    '        vp_stage,\n'
    '        config=config,\n'
    '        pg_collection=pg_collection,\n'
    '    )\n'
    '    maybe_set_structure_inputs(model, {"structure_ids": None})\n'
    '    return model\n'
)
if 'maybe_set_structure_inputs(model, {"structure_ids": None})' not in text:
    text = text.replace(needle, replacement, 1)
p.write_text(text)
PY

cat > "${REMOTE_WORKDIR}/model_provider.py" <<'PY'
from megatron.training import get_args
from cppmega.megatron.structure_batch import maybe_set_structure_inputs


def model_provider(
    model_builder,
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    args = get_args()
    model = model_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
    return model
PY

cat > "${REMOTE_WORKDIR}/structure_ingress_probe.py" <<'PY'
import torch
from cppmega.megatron.custom_embedding import CppMegaLanguageModelEmbedding

class _Config:
    hidden_size = 16
    hidden_dropout = 0.0
    sequence_parallel = False
    clone_scatter_output_in_embedding = False
    fp32_residual_connection = False
    use_mup = False
    mup_embedding_mult = 1.0
    perform_initialization = True
    use_cpu_initialization = False
    deterministic_mode = False
    params_dtype = torch.float32
    pipeline_dtype = torch.float32
    init_method = None
    @staticmethod
    def embedding_init_method(weight):
        torch.nn.init.zeros_(weight)

embedding = CppMegaLanguageModelEmbedding(
    config=_Config(),
    vocab_size=128,
    max_sequence_length=16,
    position_embedding_type='rope',
    scatter_to_sequence_parallel=False,
    tp_group=None,
)
with torch.no_grad():
    if embedding.cppmega_structure is not None:
        embedding.cppmega_structure.stacked_emb.weight.fill_(0.25)
        embedding.cppmega_structure.up_proj.weight.fill_(0.5)
input_ids = torch.randint(0, 16, (2, 4), device='cpu')
position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)
structure_inputs = {
    'structure_ids': torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long),
    'dep_levels': torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long),
}
out = embedding(input_ids, position_ids, structure_inputs=structure_inputs)
print('structure ingress probe nonzero=', int(torch.count_nonzero(out).item()))
PY

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CPPMEGA_STRUCTURE_ENABLED=1
export CPPMEGA_STRUCTURE_COMPONENTS=core
python "${REMOTE_WORKDIR}/structure_ingress_probe.py" > "${REMOTE_LOG}" 2>&1
cat "${REMOTE_LOG}"
INNER

gcloud compute scp --zone "${REMOTE_ZONE}" "${LOCAL_TMP_SCRIPT}" "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' bash '${REMOTE_TMP_SCRIPT}'"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "rm -f '${REMOTE_TMP_SCRIPT}'"
