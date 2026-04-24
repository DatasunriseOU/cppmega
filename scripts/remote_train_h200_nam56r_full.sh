#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_legacy}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_3}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_nam56r_full_train}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-nam56r-full-train.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-nam56r-full.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"
mkdir -p "${REMOTE_ROOT}/cppmega" "${REMOTE_ROOT}/.triton-cache"
REMOTE_CKPT_DIR="${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}_ckpt"
REMOTE_WORKDIR="$(mktemp -d /tmp/cppmega-nam56r-full.XXXXXX)"
trap 'rm -rf "${REMOTE_WORKDIR}"' EXIT

python -c "import cppmega, megatron, transformer_engine; print('import smoke ok', cppmega.__version__)"

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

python -m torch.distributed.run --nproc_per_node=8 "${REMOTE_WORKDIR}/pretrain_mamba.py" \
  --data-path "1.0 ${REMOTE_ROOT:-/mnt/data}/data/megatron/clang_semantic_4k_v10_train" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${REMOTE_ROOT:-/mnt/data}/tokenizer" \
  --vocab-size 131072 \
  --make-vocab-size-divisible-by 128 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size "${CPPMEGA_PP_SIZE:-4}" \
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
  --seq-length "${CPPMEGA_SEQ_LENGTH:-4096}" \
  --max-position-embeddings "${CPPMEGA_MAX_POSITION_EMBEDDINGS:-4096}" \
  --micro-batch-size "${CPPMEGA_MICRO_BATCH_SIZE:-1}" \
  --global-batch-size "${CPPMEGA_GLOBAL_BATCH_SIZE:-8}" \
  --train-iters "${CPPMEGA_TRAIN_ITERS:-2}" \
  --eval-interval 50000000 \
  --eval-iters 0 \
  --lr "${CPPMEGA_LR:-1e-4}" \
  --min-lr "${CPPMEGA_MIN_LR:-1e-5}" \
  --lr-decay-style constant \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --bf16 \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --use-flash-attn \
  --spec cppmega.megatron.nam56r_full_spec build_cppmega_nam56r_full_stack_spec \
  ${NATIVE_ARGS} \
  --save "${REMOTE_CKPT_DIR}" \
  --load "${REMOTE_CKPT_DIR}" \
  --save-interval 1 \
  --log-interval 1 \
  > "${REMOTE_LOG}" 2>&1

tail -n 120 "${REMOTE_LOG}"
INNER

gcloud compute scp --zone "${REMOTE_ZONE}" "${LOCAL_TMP_SCRIPT}" "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' CPPMEGA_TRAIN_ITERS='${CPPMEGA_TRAIN_ITERS:-2}' CPPMEGA_PP_SIZE='${CPPMEGA_PP_SIZE:-4}' CPPMEGA_SEQ_LENGTH='${CPPMEGA_SEQ_LENGTH:-4096}' CPPMEGA_MAX_POSITION_EMBEDDINGS='${CPPMEGA_MAX_POSITION_EMBEDDINGS:-4096}' CPPMEGA_MICRO_BATCH_SIZE='${CPPMEGA_MICRO_BATCH_SIZE:-1}' CPPMEGA_GLOBAL_BATCH_SIZE='${CPPMEGA_GLOBAL_BATCH_SIZE:-8}' CPPMEGA_LR='${CPPMEGA_LR:-1e-4}' CPPMEGA_MIN_LR='${CPPMEGA_MIN_LR:-1e-5}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'remote nam56r-full train failed; tail follows:'; tail -n 200 '${REMOTE_LOG}' || true; exit \$status)"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "rm -f '${REMOTE_TMP_SCRIPT}'"
