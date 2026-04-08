#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_legacy}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_3}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_dsa_smoke}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-dsa-smoke.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-dsa-smoke.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'EOF'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"
mkdir -p "${REMOTE_ROOT}/cppmega" "${REMOTE_ROOT}/.triton-cache"
REMOTE_CKPT_DIR="${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}_ckpt"

python -c "import cppmega, megatron; print('import smoke ok', cppmega.__version__)"

cd "${REMOTE_ROOT}/megatron-lm"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CPPMEGA_DSA_DISABLE_ROPE_FUSION=1
NATIVE_ARGS="$(${REMOTE_VENV}/bin/python - <<'PY'
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan

plan = build_nam56r_feature_plan(pattern='AEMEAEMEAEMR', depth=52)
bundle = build_nam56r_megatron_native_args(plan=plan, enable_mla=True, enable_dsa=True)
print(bundle.to_shell_fragment())
PY
)"

python - <<'PY'
from pathlib import Path

p = Path('/mnt/data/megatron-lm/megatron/training/arguments.py')
text = p.read_text()
needle = "    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm\n"
replacement = needle + (
    "    if getattr(args, 'experimental_attention_variant', None) == 'dsa':\n"
    "        kw_args['apply_rope_fusion'] = False\n"
)
if replacement not in text:
    if needle not in text:
        raise SystemExit('failed to find Megatron config insertion point for DSA rope fusion override')
    p.write_text(text.replace(needle, replacement, 1))

spec_path = Path('/mnt/data/megatron-lm/megatron/core/models/gpt/experimental_attention_variant_module_specs.py')
spec_text = spec_path.read_text()
spec_old = (
    '    if config.experimental_attention_variant == "gated_delta_net":\n'
    '        return get_gated_delta_net_module_spec(config=config, backend=backend)\n'
    '    else:\n'
    '        raise ValueError(\n'
    '            f"Invalid experimental attention variant: {config.experimental_attention_variant}"\n'
    '        )\n'
)
spec_new = (
    '    if config.experimental_attention_variant == "gated_delta_net":\n'
    '        return get_gated_delta_net_module_spec(config=config, backend=backend)\n'
    '    if config.experimental_attention_variant == "dsa":\n'
    '        return get_dsa_module_spec_for_backend(config=config, backend=backend)\n'
    '    raise ValueError(\n'
    '        f"Invalid experimental attention variant: {config.experimental_attention_variant}"\n'
    '    )\n'
)
if spec_new not in spec_text:
    if spec_old not in spec_text:
        raise SystemExit('failed to find Megatron experimental attention variant routing block for DSA patch')
    spec_path.write_text(spec_text.replace(spec_old, spec_new, 1))

layernorm_old = '            if attention.metainfo["fuse_input_layernorm"]\n'
layernorm_new = '            if attention.metainfo.get("fuse_input_layernorm", False)\n'
spec_text = spec_path.read_text()
if layernorm_new not in spec_text:
    if layernorm_old not in spec_text:
        raise SystemExit('failed to find Megatron DSA input-layernorm metainfo access for smoke patch')
    spec_path.write_text(spec_text.replace(layernorm_old, layernorm_new, 1))
PY

python -m torch.distributed.run --nproc_per_node=8 pretrain_gpt.py \
  --mock-data \
  --tokenizer-type NullTokenizer \
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
  ${NATIVE_ARGS} \
  --save "${REMOTE_CKPT_DIR}" \
  --load "${REMOTE_CKPT_DIR}" \
  --save-interval 1 \
  --log-interval 1 \
  > "${REMOTE_LOG}" 2>&1

tail -n 60 "${REMOTE_LOG}"
EOF

gcloud compute scp \
  --zone "${REMOTE_ZONE}" \
  "${LOCAL_TMP_SCRIPT}" \
  "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'remote dsa smoke failed; tail follows:'; tail -n 100 '${REMOTE_LOG}' || true; exit \$status)"

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "rm -f '${REMOTE_TMP_SCRIPT}'"
