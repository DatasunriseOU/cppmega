#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_legacy}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_3}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_nam56r_mixed_a_smoke}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-nam56r-mixed-a-smoke.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-nam56r-mixed-a.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"
mkdir -p "${REMOTE_ROOT}/cppmega" "${REMOTE_ROOT}/.triton-cache"
REMOTE_CKPT_DIR="${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}_ckpt"
REMOTE_WORKDIR="$(mktemp -d /tmp/cppmega-nam56r-mixed-a.XXXXXX)"
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
export CPPMEGA_DSA_A_LAYER_RANKS="${CPPMEGA_DSA_A_LAYER_RANKS:-8,9,10,11}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

HYBRID_LAYER_PATTERN="$(${REMOTE_VENV}/bin/python - <<'PY'
from cppmega.megatron.nam56r_full_spec import build_default_hybrid_layer_pattern
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
    enable_dsa=True,
)
print(bundle.to_shell_fragment())
PY
)"

python -m torch.distributed.run --nproc_per_node=8 "${REMOTE_WORKDIR}/pretrain_mamba.py" \
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
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}" \
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
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' CPPMEGA_DSA_A_LAYER_RANKS='${CPPMEGA_DSA_A_LAYER_RANKS:-8,9,10,11}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'remote nam56r mixed-A smoke failed; tail follows:'; tail -n 160 '${REMOTE_LOG}' || true; exit \$status)"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "rm -f '${REMOTE_TMP_SCRIPT}'"
