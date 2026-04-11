#!/usr/bin/env bash
# Launch NAM56R training on H200 using the vanilla Mamba-2 SSD kernel path
# with Mamba3 B/C features (QK-Norm + learnable bias) injected as Python
# preprocessing.  This is Branch B from the Test-loop task -- an alternative
# to the mamba3_te recipe (Author kernels, 127k tok/sec) that reuses the
# proven-fast mamba_chunk_scan_combined kernel.
#
# Usage (same contract as remote_train_h200_nam56r_full.sh):
#
#     REMOTE_HOST=h200_legacy \
#     REMOTE_ZONE=LOCATION_3 \
#     CPPMEGA_TRAIN_ITERS=10 \
#     bash scripts/remote_train_h200_nam56r_noconv.sh
#
# The script is intentionally a near-copy of remote_train_h200_nam56r_full.sh
# so differences are visible in diff view: the ONLY change is the ``--spec``
# argument to point at the noconv stack builder.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_legacy}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_3}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_nam56r_noconv_train}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-nam56r-noconv-train.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-nam56r-noconv.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"
# Force TransformerEngine to load the venv's cuDNN 9.20 (which has MLA engine configs
# for sm_90 at head_dim_qk=96/head_dim_v=64), NOT the system cuDNN 9.10.2 that ldconfig
# resolves by default. Without this, the unversioned ctypes.CDLL("libcudnn.so") in TE's
# _load_cuda_library falls through to /lib/x86_64-linux-gnu which has the older version
# that lacks MLA configs and crashes with "No valid engine configs" on the NAM56R attn
# layer. A sibling fix creates the venv's libcudnn.so -> libcudnn.so.9 symlink.
_VENV_CUDNN_LIB="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/cudnn/lib"
if [ -d "${_VENV_CUDNN_LIB}" ]; then
  export LD_LIBRARY_PATH="${_VENV_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi
# Reduce PyTorch CUDA memory fragmentation. CUDA graph capture pre-allocates
# large static workspaces; without expandable_segments the allocator fragments
# badly on MBS=4 + MTP + MoE16-top4 + grouped GEMM and OOMs even though the
# 141 GiB H200 has plenty of raw capacity. Observed on smoke6/7 where
# GPU used=72.7 GiB, fragmented reserved=1.5 GiB, failed to alloc 516 MiB.
# expandable_segments lets the allocator remap/coalesce free chunks so CUDA
# graph workspace + per-step buffers can share the same heap.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# expandable_segments:True + CUDA graph capture requires NCCL_GRAPH_REGISTER=0
# to avoid illegal memory access when NCCL tries to register graph-captured
# buffers that get remapped by the expandable allocator. Megatron asserts this
# combination at arguments.py:1573-1576.
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}"
mkdir -p "${REMOTE_ROOT}/cppmega" "${REMOTE_ROOT}/.triton-cache"
REMOTE_CKPT_DIR="${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}_ckpt"
REMOTE_WORKDIR="$(mktemp -d /tmp/cppmega-nam56r-noconv.XXXXXX)"
trap 'rm -rf "${REMOTE_WORKDIR}"' EXIT

python -c "import cppmega, megatron, transformer_engine; print('import smoke ok', cppmega.__version__)"
python -c "from cppmega.megatron.nam56r_noconv_spec import build_cppmega_nam56r_noconv_stack_spec; print('noconv spec importable')"

cp "${REMOTE_ROOT}/megatron-lm/pretrain_mamba.py" "${REMOTE_WORKDIR}/pretrain_mamba.py"

cat > "${REMOTE_WORKDIR}/mamba_builders.py" <<'PY'
from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder
PY

cat > "${REMOTE_WORKDIR}/model_provider.py" <<'PY'
from megatron.training import get_args


def model_provider(
    model_builder,
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    args = get_args()
    return model_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
PY

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
# CUDA graph capture with TE requires NCCL_GRAPH_REGISTER=0 to avoid illegal
# memory access when PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is also
# active (which Megatron sets by default). Without this, training crashes at
# startup with "AssertionError: Setting NCCL_GRAPH_REGISTER=0 to avoid illegal
# memory access when using CUDA Graph".
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}"
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

HYBRID_LAYER_PATTERN="$(${REMOTE_VENV}/bin/python - <<'PY'
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern
print(build_default_hybrid_layer_pattern(mtp_depths=1))
PY
)"

NATIVE_ARGS="$(${REMOTE_VENV}/bin/python - <<'PY'
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan

plan = build_nam56r_feature_plan(pattern='AEMEAEMEAEMR', depth=52, mtp_depths=1)
bundle = build_nam56r_megatron_native_args(
    plan=plan,
    enable_mla=True,
    enable_mtp=True,
    mtp_mode='hybrid',
    enable_moe=True,
)
print(bundle.to_shell_fragment())
PY
)"

if [ -n "${CPPMEGA_DATA_PATH:-}" ]; then
  DATA_ARGS=(--data-path ${CPPMEGA_DATA_PATH})
  DATA_ARGS+=(--split "${CPPMEGA_DATA_SPLIT:-98,1,1}")
  if [ -n "${CPPMEGA_TOKENIZER_MODEL:-}" ]; then
    DATA_ARGS+=(
      --tokenizer-type "${CPPMEGA_TOKENIZER_TYPE:-HuggingFaceTokenizer}"
      --tokenizer-model "${CPPMEGA_TOKENIZER_MODEL}"
    )
  else
    DATA_ARGS+=(
      --tokenizer-type NullTokenizer
      --vocab-size "${CPPMEGA_VOCAB_SIZE:-131072}"
      --make-vocab-size-divisible-by 128
    )
  fi
else
  DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 131072
    --make-vocab-size-divisible-by 128
  )
fi

# CUDA graph capture wiring.
#
# NAM56R uses dropless MoE (no moe-expert-capacity-factor and no
# moe-pad-expert-input-to-capacity) so the MoE layer has dynamic shapes and
# cannot be captured in a CUDA graph -- attempting to do so hits
# "RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
# capture unless the CPU tensor is pinned" inside MoEAlltoAll dispatcher's
# tokens_per_expert = self.local_map.sum(dim=0).long().cpu() on line 295.
#
# Per megatron-core/transformer/moe/README.md, the correct config for dropless
# MoE with CUDA graphs is:
#   --cuda-graph-impl=transformer_engine --cuda-graph-scope=attn
# which captures only TransformerLayer._forward_attention() and leaves the MoE
# _forward_mlp() untouched. Also required: --te-rng-tracker.
#
# The --cuda-graph-scope=attn scope is NOT supported by --cuda-graph-impl=local
# (local only supports full_iteration / full_iteration_inference scopes).
#
# Avoid --moe-shared-expert-overlap (incompatible with graphs).
#
# Env vars:
#   CPPMEGA_CUDA_GRAPH       = te_attn (default) | te_full | local_full | none
#     te_attn   -> TE impl, attn-only scope  (works with dropless MoE)
#     te_full   -> TE impl, full scope       (likely hits MoE shared expert assert)
#     local_full-> local impl, full_iteration scope (local+dropless MoE fails)
#     none      -> CUDA graphs disabled
#   CPPMEGA_CUDA_GRAPH_WARMUP_STEPS = integer warmup steps before capture
CPPMEGA_CUDA_GRAPH="${CPPMEGA_CUDA_GRAPH:-te_attn}"
CUDA_GRAPH_ARGS=()
case "${CPPMEGA_CUDA_GRAPH}" in
  te_attn)
    # te_rng_tracker is auto-enabled by Megatron when cuda_graph_impl includes
    # transformer_engine (arguments.py:1568-1571), no need to pass it as CLI flag
    # (there is no --te-rng-tracker argparse declaration anyway).
    CUDA_GRAPH_ARGS+=(
      --cuda-graph-impl transformer_engine
      --cuda-graph-scope attn
      --cuda-graph-warmup-steps "${CPPMEGA_CUDA_GRAPH_WARMUP_STEPS:-3}"
    )
    ;;
  te_full)
    CUDA_GRAPH_ARGS+=(
      --cuda-graph-impl transformer_engine
      --cuda-graph-warmup-steps "${CPPMEGA_CUDA_GRAPH_WARMUP_STEPS:-3}"
    )
    ;;
  local_full)
    CUDA_GRAPH_ARGS+=(
      --cuda-graph-impl local
      --cuda-graph-scope full_iteration
      --cuda-graph-warmup-steps "${CPPMEGA_CUDA_GRAPH_WARMUP_STEPS:-3}"
      --no-check-for-nan-in-loss-and-grad
    )
    ;;
  none)
    CUDA_GRAPH_ARGS+=(--cuda-graph-impl none)
    ;;
  *)
    echo "Unknown CPPMEGA_CUDA_GRAPH=${CPPMEGA_CUDA_GRAPH}; expected te_attn|te_full|local_full|none" >&2
    exit 2
    ;;
esac

# FP8 support.
#   CPPMEGA_FP8=1     → FP8 hybrid format (per-tensor current scaling, TE 2.13+)
#   CPPMEGA_FP8=0     → BF16 (default)
# FP8 requires nheads % 16 == 0 (use CPPMEGA_MAMBA_NUM_HEADS=64 for FP8 mode).
FP8_ARGS=()
if [ "${CPPMEGA_FP8:-0}" = "1" ]; then
  FP8_ARGS+=(
    --fp8-format hybrid
    --fp8-amax-history-len 16
    --fp8-amax-compute-algo max
  )
fi

# Precision: FP8 mode overrides --bf16 with FP8 args; otherwise BF16.
PRECISION_ARGS=()
if [ "${CPPMEGA_FP8:-0}" = "1" ]; then
  PRECISION_ARGS+=(--bf16)  # base precision for non-FP8 ops
else
  PRECISION_ARGS+=(--bf16)
fi

python -m torch.distributed.run --nproc_per_node=8 "${REMOTE_WORKDIR}/pretrain_mamba.py" \
  "${DATA_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size "${CPPMEGA_PP_SIZE:-1}" \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}" \
  --hidden-size 3584 \
  --ffn-hidden-size 18944 \
  --num-attention-heads "${CPPMEGA_NUM_ATTN_HEADS:-28}" \
  --seq-length "${CPPMEGA_SEQ_LENGTH:-4096}" \
  --max-position-embeddings "${CPPMEGA_MAX_POSITION_EMBEDDINGS:-4096}" \
  --micro-batch-size "${CPPMEGA_MICRO_BATCH_SIZE:-1}" \
  --global-batch-size "${CPPMEGA_GLOBAL_BATCH_SIZE:-8}" \
  --train-iters "${CPPMEGA_TRAIN_ITERS:-2}" \
  --eval-interval 50000000 \
  --eval-iters "${CPPMEGA_EVAL_ITERS:-1}" \
  --lr "${CPPMEGA_LR:-1e-4}" \
  --min-lr "${CPPMEGA_MIN_LR:-1e-5}" \
  --lr-decay-style constant \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  "${PRECISION_ARGS[@]}" \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --attention-backend "${CPPMEGA_ATTN_BACKEND:-auto}" \
  --spec cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${FP8_ARGS[@]}" \
  ${NATIVE_ARGS} \
  --save "${REMOTE_CKPT_DIR}" \
  --load "${REMOTE_CKPT_DIR}" \
  --save-interval "${CPPMEGA_SAVE_INTERVAL:-1}" \
  --log-interval 1 \
  > "${REMOTE_LOG}" 2>&1

tail -n 120 "${REMOTE_LOG}"
INNER

gcloud compute scp --zone "${REMOTE_ZONE}" "${LOCAL_TMP_SCRIPT}" "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' CPPMEGA_TRAIN_ITERS='${CPPMEGA_TRAIN_ITERS:-2}' CPPMEGA_PP_SIZE='${CPPMEGA_PP_SIZE:-4}' CPPMEGA_SEQ_LENGTH='${CPPMEGA_SEQ_LENGTH:-4096}' CPPMEGA_MAX_POSITION_EMBEDDINGS='${CPPMEGA_MAX_POSITION_EMBEDDINGS:-4096}' CPPMEGA_MICRO_BATCH_SIZE='${CPPMEGA_MICRO_BATCH_SIZE:-1}' CPPMEGA_GLOBAL_BATCH_SIZE='${CPPMEGA_GLOBAL_BATCH_SIZE:-8}' CPPMEGA_LR='${CPPMEGA_LR:-1e-4}' CPPMEGA_MIN_LR='${CPPMEGA_MIN_LR:-1e-5}' CPPMEGA_DATA_PATH='${CPPMEGA_DATA_PATH:-}' CPPMEGA_DATA_SPLIT='${CPPMEGA_DATA_SPLIT:-98,1,1}' CPPMEGA_TOKENIZER_TYPE='${CPPMEGA_TOKENIZER_TYPE:-HuggingFaceTokenizer}' CPPMEGA_TOKENIZER_MODEL='${CPPMEGA_TOKENIZER_MODEL:-}' CPPMEGA_VOCAB_SIZE='${CPPMEGA_VOCAB_SIZE:-131072}' CPPMEGA_EVAL_ITERS='${CPPMEGA_EVAL_ITERS:-1}' CPPMEGA_SAVE_INTERVAL='${CPPMEGA_SAVE_INTERVAL:-1}' CPPMEGA_ATTN_BACKEND='${CPPMEGA_ATTN_BACKEND:-auto}' CPPMEGA_CUDA_GRAPH='${CPPMEGA_CUDA_GRAPH:-te_attn}' CPPMEGA_CUDA_GRAPH_WARMUP_STEPS='${CPPMEGA_CUDA_GRAPH_WARMUP_STEPS:-3}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'remote nam56r-noconv train failed; tail follows:'; tail -n 200 '${REMOTE_LOG}' || true; exit \$status)"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "rm -f '${REMOTE_TMP_SCRIPT}'"
