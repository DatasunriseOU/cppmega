#!/usr/bin/env bash
# Full NAM56R + DSA on 8xH200 with real data and the FP8 indexer path.
#
# This is the Stream E variant of scripts/remote_smoke_h200_dsa_full_nam56r.sh:
# it keeps the BF16 baseline exactly intact but sets CPPMEGA_DSA_INDEXER_DTYPE=fp8
# and prepends cppmega.megatron.dsa_fp8_patch.apply_dsa_fp8_patch() to the
# pretrain_mamba.py launcher so Megatron's private `_compute_index_scores` is
# monkey-patched to the cppmega FP8 variant (port of DeepSeek V3.2
# inference/kernel.py fp8_index) before any model module is built.
#
# Usage:
#   REMOTE_HOST=h200_1 \
#   REMOTE_ZONE=LOCATION_1 \
#   bash scripts/remote_smoke_h200_dsa_fp8_indexer.sh
#
# Expected comparison:
#   * BF16 baseline from scripts/remote_smoke_h200_dsa_full_nam56r.sh
#   * FP8 variant from this script, same layer layout, same micro/global batch
#   * Diff logged in docs/nam56r_grid_search_2026_04_12.md
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_1}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_1}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_nam56r_dsa_full_noconv_fp8_indexer}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega-root/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-dsa-fp8-indexer.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-dsa-fp8.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'EOF'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega-root/cppmega:${REMOTE_ROOT}/cppmega-root/megatron-lm:${PYTHONPATH:-}"

# Force venv cuDNN for MLA engine configs
_VENV_CUDNN_LIB="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/cudnn/lib"
if [ -d "${_VENV_CUDNN_LIB}" ]; then
  export LD_LIBRARY_PATH="${_VENV_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}"
mkdir -p "${REMOTE_ROOT}/cppmega-root/cppmega" "${REMOTE_ROOT}/.triton-cache"
REMOTE_CKPT_DIR="${REMOTE_ROOT}/cppmega-root/cppmega/${CPPMEGA_RUN_ID}_ckpt"

python -c "import cppmega, megatron; print('import smoke ok', cppmega.__version__)"
python -c "from cppmega.megatron.nam56r_noconv_spec import build_cppmega_nam56r_noconv_stack_spec; print('noconv spec importable')"
python -c "from cppmega.megatron.dsa_fp8_indexer import compute_index_scores_fp8; print('dsa_fp8_indexer importable')"
python -c "from cppmega.megatron.dsa_fp8_patch import apply_dsa_fp8_patch; print('dsa_fp8_patch importable')"

WORKDIR="$(mktemp -d /tmp/cppmega-dsa-fp8.XXXXXX)"
trap 'rm -rf "${WORKDIR}"' EXIT

# Wrap pretrain_mamba.py so the FP8 monkey-patch runs before any model build.
cat > "${WORKDIR}/pretrain_mamba.py" <<'PY'
"""Stream E FP8 DSA indexer wrapper around Megatron's pretrain_mamba.py.

Applies the cppmega FP8 DSA indexer monkey-patch *before* Megatron imports
pull in the DSA module, then execs the upstream pretrain_mamba.py in the
current module namespace so the launcher arguments behave identically to
the BF16 baseline.
"""

import os

os.environ.setdefault("CPPMEGA_DSA_INDEXER_DTYPE", "fp8")

from cppmega.megatron.dsa_fp8_patch import apply_dsa_fp8_patch

_patched = apply_dsa_fp8_patch()
print(f"[cppmega] dsa_fp8_patch applied={_patched} CPPMEGA_DSA_INDEXER_DTYPE={os.environ.get('CPPMEGA_DSA_INDEXER_DTYPE', 'bf16')}", flush=True)

with open("${REMOTE_ROOT}/cppmega-root/megatron-lm/pretrain_mamba.py", "r") as f:
    _src = f.read()
exec(compile(_src, "pretrain_mamba.py", "exec"), globals())
PY

cat > "${WORKDIR}/mamba_builders.py" <<'PY'
from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder
PY

cat > "${WORKDIR}/model_provider.py" <<'PY'
from megatron.training import get_args

def model_provider(model_builder, pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    args = get_args()
    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)
PY

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CPPMEGA_DSA_DISABLE_ROPE_FUSION=1
export CPPMEGA_DSA_INDEXER_DTYPE=fp8
export CPPMEGA_NEM_PATTERN="AEMEAEMEAEMR"
export CPPMEGA_LAYER_DEPTH=52
export CPPMEGA_R_LAYER_INDICES="12,24,36,48"
export CPPMEGA_NGRAM_HASH_ENABLED="${CPPMEGA_NGRAM_HASH_ENABLED:-1}"
export CPPMEGA_NGRAM_HASH_ORDERS="${CPPMEGA_NGRAM_HASH_ORDERS:-2,3}"
export CPPMEGA_NGRAM_HASH_HEADS="${CPPMEGA_NGRAM_HASH_HEADS:-8}"
export CPPMEGA_NGRAM_HASH_TABLE_SIZE="${CPPMEGA_NGRAM_HASH_TABLE_SIZE:-500000}"
export CPPMEGA_NGRAM_HASH_EMBED_DIM="${CPPMEGA_NGRAM_HASH_EMBED_DIM:-16}"
export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-1}"
export CPPMEGA_STRUCTURE_COMPONENTS="${CPPMEGA_STRUCTURE_COMPONENTS:-core}"

HYBRID_LAYER_PATTERN="$(${REMOTE_VENV}/bin/python -c "
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern
print(build_default_hybrid_layer_pattern(mtp_depths=1))
")"

NATIVE_ARGS="$(${REMOTE_VENV}/bin/python -c "
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan
plan = build_nam56r_feature_plan(pattern='AEMEAEMEAEMR', depth=52, mtp_depths=1)
bundle = build_nam56r_megatron_native_args(
    plan=plan,
    enable_mla=True,
    enable_mtp=True,
    mtp_mode='hybrid',
    enable_moe=True,
    enable_dsa=True,
    dsa_indexer_dtype='fp8',
)
print(bundle.to_shell_fragment())
")"

# Apply DSA Megatron patches (idempotent) — reuse the same set the BF16 lane
# applies, so Stream D's baseline and this FP8 variant share an identical
# patched Megatron checkout on bench3.
python - <<'PY'
from pathlib import Path

# Patch 1: disable rope fusion for DSA
p = Path('${REMOTE_ROOT}/cppmega-root/megatron-lm/megatron/training/arguments.py')
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
    print('DSA rope fusion patch applied')
else:
    print('DSA rope fusion patch already present')

# Patch 2: add DSA routing in experimental attention variant specs
spec_path = Path('${REMOTE_ROOT}/cppmega-root/megatron-lm/megatron/core/models/gpt/experimental_attention_variant_module_specs.py')
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
    print('DSA routing patch applied')
else:
    print('DSA routing patch already present')

# Patch 3: fix fuse_input_layernorm metainfo access
layernorm_old = '            if attention.metainfo["fuse_input_layernorm"]\n'
layernorm_new = '            if attention.metainfo.get("fuse_input_layernorm", False)\n'
spec_text = spec_path.read_text()
if layernorm_new not in spec_text:
    if layernorm_old not in spec_text:
        raise SystemExit('failed to find Megatron DSA input-layernorm metainfo access for smoke patch')
    spec_path.write_text(spec_text.replace(layernorm_old, layernorm_new, 1))
    print('DSA layernorm metainfo patch applied')
else:
    print('DSA layernorm metainfo patch already present')
PY

python -m torch.distributed.run --nproc_per_node=8 "${WORKDIR}/pretrain_mamba.py" \
  --data-path "1.0" "${REMOTE_ROOT}/data/megatron/clang_semantic_4k_v10_train" \
  --split 98,1,1 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${REMOTE_ROOT}/tokenizer" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}" \
  --hidden-size 3584 \
  --ffn-hidden-size 18944 \
  --num-attention-heads 28 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --micro-batch-size "${CPPMEGA_MICRO_BATCH_SIZE:-2}" \
  --global-batch-size "${CPPMEGA_GLOBAL_BATCH_SIZE:-16}" \
  --train-iters "${CPPMEGA_TRAIN_ITERS:-20}" \
  --eval-interval 50000000 \
  --eval-iters 1 \
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
  --attention-backend auto \
  --spec cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec \
  --cuda-graph-impl transformer_engine \
  --cuda-graph-scope attn \
  --cuda-graph-warmup-steps 3 \
  ${NATIVE_ARGS} \
  --save "${REMOTE_CKPT_DIR}" \
  --load "${REMOTE_CKPT_DIR}" \
  --save-interval 1000000 \
  --log-interval 1 \
  > "${REMOTE_LOG}" 2>&1

tail -60 "${REMOTE_LOG}"
EOF

gcloud compute scp \
  --zone "${REMOTE_ZONE}" \
  "${LOCAL_TMP_SCRIPT}" \
  "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' CPPMEGA_TRAIN_ITERS='${CPPMEGA_TRAIN_ITERS:-20}' CPPMEGA_MICRO_BATCH_SIZE='${CPPMEGA_MICRO_BATCH_SIZE:-2}' CPPMEGA_GLOBAL_BATCH_SIZE='${CPPMEGA_GLOBAL_BATCH_SIZE:-16}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'remote DSA FP8 indexer run failed; tail follows:'; tail -n 100 '${REMOTE_LOG}' || true; exit \$status)"

gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "rm -f '${REMOTE_TMP_SCRIPT}'"
