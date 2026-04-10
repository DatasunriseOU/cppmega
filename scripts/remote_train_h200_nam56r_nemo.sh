#!/usr/bin/env bash
# NeMo 3 Nano-style NAM56R training on 8×H200.
#
# This script follows Nemotron Nano v2 training patterns for maximum throughput:
# - Distributed optimizer with gradient overlap
# - All kernel fusions enabled (no --no-* flags)
# - Proper batch sizing for H200 memory
# - --log-throughput for MFU/tok/s measurement
#
# Two modes (set CPPMEGA_MODE):
#   nemo_native  - TP=2, SP=True, Megatron built-in Mamba (highest throughput)
#   author_dp    - TP=1, DP=8, Author Mamba3 + M²RNN (full NAM56R features)
#
# Usage:
#   CPPMEGA_MODE=nemo_native bash scripts/remote_train_h200_nam56r_nemo.sh
#   CPPMEGA_MODE=author_dp CPPMEGA_TRAIN_ITERS=100 bash scripts/remote_train_h200_nam56r_nemo.sh
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_1}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_2}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
CPPMEGA_MODE="${CPPMEGA_MODE:-nemo_native}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_nam56r_nemo_${CPPMEGA_MODE}}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-nam56r-nemo.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-nam56r-nemo.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"
mkdir -p "${REMOTE_ROOT}/cppmega" "${REMOTE_ROOT}/.triton-cache"
REMOTE_CKPT_DIR="${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}_ckpt"
REMOTE_WORKDIR="$(mktemp -d /tmp/cppmega-nam56r-nemo.XXXXXX)"
trap 'rm -rf "${REMOTE_WORKDIR}"' EXIT

# ---- Import smoke ----
python -c "import cppmega, megatron; print('imports ok')"

# ---- Copy pretrain entry point ----
cp "${REMOTE_ROOT}/megatron-lm/pretrain_mamba.py" "${REMOTE_WORKDIR}/pretrain_mamba.py"

# ---- Create builder shim ----
if [ "${CPPMEGA_MODE}" = "author_dp" ]; then
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
else
  # nemo_native: use standard Megatron mamba builder + model_provider
  cp "${REMOTE_ROOT}/megatron-lm/mamba_builders.py" "${REMOTE_WORKDIR}/mamba_builders.py"
  cp "${REMOTE_ROOT}/megatron-lm/model_provider.py" "${REMOTE_WORKDIR}/model_provider.py"
fi

# ---- Environment variables (NeMo-style) ----
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# NeMo kernel optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System now has CUDA 13.2 + cuDNN 9.20 + cuBLAS 13.3. No LD_LIBRARY_PATH needed.

# cppmega pattern config
export CPPMEGA_NEM_PATTERN="${CPPMEGA_NEM_PATTERN:-AEMEAEMEAEMR}"
export CPPMEGA_LAYER_DEPTH="${CPPMEGA_LAYER_DEPTH:-52}"
export CPPMEGA_R_LAYER_INDICES="${CPPMEGA_R_LAYER_INDICES:-12,24,36,48}"

# ---- Generate recipe args via Python ----
RECIPE_ARGS="$(python - <<PYEOF
from cppmega.recipes.nam56r_nemo_recipe import (
    nam56r_nemo_native_pretrain,
    nam56r_author_dp_pretrain,
)
import os

mode = os.environ.get("CPPMEGA_MODE", "nemo_native")
if mode == "nemo_native":
    recipe = nam56r_nemo_native_pretrain()
elif mode == "author_dp":
    recipe = nam56r_author_dp_pretrain()
else:
    raise ValueError(f"unsupported CPPMEGA_MODE: {mode}")

# Apply env overrides
import dataclasses
overrides = {}
train_iters = os.environ.get("CPPMEGA_TRAIN_ITERS")
if train_iters:
    overrides["train_iters"] = int(train_iters)
micro_batch = os.environ.get("CPPMEGA_MICRO_BATCH_SIZE")
if micro_batch:
    overrides["micro_batch_size"] = int(micro_batch)
global_batch = os.environ.get("CPPMEGA_GLOBAL_BATCH_SIZE")
if global_batch:
    overrides["global_batch_size"] = int(global_batch)
seq_len = os.environ.get("CPPMEGA_SEQ_LENGTH")
if seq_len:
    overrides["seq_length"] = int(seq_len)
    overrides["max_position_embeddings"] = int(seq_len)
lr = os.environ.get("CPPMEGA_LR")
if lr:
    overrides["lr"] = float(lr)
data_path = os.environ.get("CPPMEGA_DATA_PATH")
if data_path:
    overrides["mock_data"] = False
    overrides["data_path"] = data_path
    overrides["tokenizer_type"] = os.environ.get("CPPMEGA_TOKENIZER_TYPE", "GPTSentencePieceTokenizer")
    overrides["tokenizer_model"] = os.environ.get("CPPMEGA_TOKENIZER_MODEL", "")
save_dir = os.environ.get("CPPMEGA_SAVE_DIR")
if save_dir:
    overrides["save_dir"] = save_dir
    overrides["load_dir"] = save_dir
transformer_impl = os.environ.get("CPPMEGA_TRANSFORMER_IMPL")
if transformer_impl:
    overrides["transformer_impl"] = transformer_impl

if overrides:
    recipe = dataclasses.replace(recipe, **overrides)

# Print shell-safe args (one per line for readability in log)
print(" ".join(recipe.to_args()))
PYEOF
)"

echo "=== NAM56R NeMo Training ==="
echo "mode: ${CPPMEGA_MODE}"
echo "run_id: ${CPPMEGA_RUN_ID}"
echo "recipe args generated"

# ---- Launch training ----
python -m torch.distributed.run \
  --nproc_per_node=8 \
  "${REMOTE_WORKDIR}/pretrain_mamba.py" \
  ${RECIPE_ARGS} \
  > "${REMOTE_LOG}" 2>&1

echo "=== Training complete ==="
echo "log: ${REMOTE_LOG}"

# ---- Print throughput summary ----
echo ""
echo "=== Throughput Summary ==="
grep -E "(throughput|tps|MFU|tokens.per.sec|elapsed)" "${REMOTE_LOG}" | tail -20 || true
echo ""
echo "=== Last 30 lines ==="
tail -n 30 "${REMOTE_LOG}"
INNER

gcloud compute scp --zone "${REMOTE_ZONE}" "${LOCAL_TMP_SCRIPT}" "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"
gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' CPPMEGA_MODE='${CPPMEGA_MODE}' CPPMEGA_TRAIN_ITERS='${CPPMEGA_TRAIN_ITERS:-100}' CPPMEGA_MICRO_BATCH_SIZE='${CPPMEGA_MICRO_BATCH_SIZE:-}' CPPMEGA_GLOBAL_BATCH_SIZE='${CPPMEGA_GLOBAL_BATCH_SIZE:-}' CPPMEGA_SEQ_LENGTH='${CPPMEGA_SEQ_LENGTH:-}' CPPMEGA_LR='${CPPMEGA_LR:-}' CPPMEGA_DATA_PATH='${CPPMEGA_DATA_PATH:-}' CPPMEGA_TOKENIZER_TYPE='${CPPMEGA_TOKENIZER_TYPE:-}' CPPMEGA_TOKENIZER_MODEL='${CPPMEGA_TOKENIZER_MODEL:-}' CPPMEGA_SAVE_DIR='${CPPMEGA_SAVE_DIR:-}' CPPMEGA_TRANSFORMER_IMPL='${CPPMEGA_TRANSFORMER_IMPL:-}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'training failed; tail follows:'; tail -n 200 '${REMOTE_LOG}' 2>/dev/null || true; exit \$status)"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "rm -f '${REMOTE_TMP_SCRIPT}'"
