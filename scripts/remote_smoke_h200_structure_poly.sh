#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_legacy}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_3}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_structure_poly_smoke}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-structure-poly-smoke.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-structure-poly.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"
mkdir -p "${REMOTE_ROOT}/cppmega" "${REMOTE_ROOT}/.triton-cache"
REMOTE_CKPT_DIR="${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}_ckpt"
REMOTE_WORKDIR="$(mktemp -d /tmp/cppmega-structure-poly.XXXXXX)"
export REMOTE_WORKDIR
trap 'rm -rf "${REMOTE_WORKDIR}"' EXIT

python -c "import cppmega, megatron; print('import smoke ok', cppmega.__version__)"

cp "${REMOTE_ROOT}/megatron-lm/pretrain_gpt.py" "${REMOTE_WORKDIR}/pretrain_gpt.py"

python - <<'PY'
from pathlib import Path

p = Path('/mnt/data/megatron-lm/model_provider.py')
text = p.read_text()
old = 'from megatron.core.models.gpt import GPTModel\n'
new = old + 'from cppmega.megatron.gpt_builder import cppmega_gpt_builder\n'
if 'from cppmega.megatron.gpt_builder import cppmega_gpt_builder\n' not in text:
    p.write_text(text.replace(old, new, 1))

needle = '    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)\n'
replacement = (
    '    if getattr(args, "position_embedding_type", None) in {"rope", "none", "learned_absolute"} and not getattr(args, "use_legacy_models", False):\n'
    '        return cppmega_gpt_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)\n'
    '    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)\n'
)
if 'return cppmega_gpt_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)' not in text:
    p.write_text(p.read_text().replace(needle, replacement, 1))
PY

python - <<'PY'
from pathlib import Path

path = Path(__import__('os').environ['REMOTE_WORKDIR']) / 'pretrain_gpt.py'
text = path.read_text()
if 'from cppmega.megatron.structure_batch import maybe_set_structure_inputs' not in text:
    text = text.replace(
        'from model_provider import model_provider\n',
        'from model_provider import model_provider\nfrom cppmega.megatron.structure_batch import maybe_set_structure_inputs\n',
        1,
    )
needle = '        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)\n'
replacement = (
    '        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)\n'
    '        structure_batch = {\n'
    '            "structure_ids": (tokens % 7).to(dtype=tokens.dtype),\n'
    '            "dep_levels": (position_ids % 5).to(dtype=position_ids.dtype),\n'
    '        }\n'
    '        maybe_set_structure_inputs(model, structure_batch)\n'
)
if 'maybe_set_structure_inputs(model, structure_batch)' not in text:
    text = text.replace(needle, replacement, 1)
path.write_text(text)
PY

cd "${REMOTE_ROOT}/megatron-lm"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CPPMEGA_STRUCTURE_ENABLED=1
export CPPMEGA_STRUCTURE_COMPONENTS="core"

python -m torch.distributed.run --nproc_per_node=8 "${REMOTE_WORKDIR}/pretrain_gpt.py" \
  --data-path "1.0 ${REMOTE_ROOT:-/mnt/data}/data/megatron/clang_semantic_4k_v10_train" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${REMOTE_ROOT:-/mnt/data}/tokenizer" \
  --vocab-size 65536 \
  --make-vocab-size-divisible-by 128 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --no-gradient-accumulation-fusion \
  --num-layers 4 \
  --hidden-size 256 \
  --ffn-hidden-size 1024 \
  --num-attention-heads 4 \
  --seq-length 128 \
  --max-position-embeddings 128 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --train-iters 2 \
  --eval-interval 50000000 \
  --eval-iters 0 \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --lr-decay-style constant \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --bf16 \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --save "${REMOTE_CKPT_DIR}" \
  --load "${REMOTE_CKPT_DIR}" \
  --save-interval 1 \
  --log-interval 1 \
  > "${REMOTE_LOG}" 2>&1

tail -n 60 "${REMOTE_LOG}"
INNER

gcloud compute scp --zone "${REMOTE_ZONE}" "${LOCAL_TMP_SCRIPT}" "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'remote structure poly smoke failed; tail follows:'; tail -n 120 '${REMOTE_LOG}' || true; exit \$status)"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "rm -f '${REMOTE_TMP_SCRIPT}'"
