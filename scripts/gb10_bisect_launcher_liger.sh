#!/usr/bin/env bash
# GB10 NAM56R 13-layer bisect launcher WITH Liger CE patches (MTP + main-head).
#
# Mirrors bench3 production cppmega_mimo_shim.py invocations of:
#   - patch_mtp_loss_with_liger()
#   - patch_mamba_output_layer_with_linear_ce()
# plus --cross-entropy-loss-fusion --cross-entropy-fusion-impl linear flag.
#
# Why: bisect on GB10 with 13-layer/52-layer/FP8/MBS=8 NEVER reproduced bench3 NaN.
# Hypothesis: bench3 NaN is in the Liger CE backward path, which the prior bisect
# launcher never installed. See docs/gb10_nan_bisect_log.md "REVISED conclusion".
#
# Run:
#   scp scripts/gb10_bisect_launcher_liger.sh gb10:/home/dave/
#   ssh gb10 'chmod +x /home/dave/gb10_bisect_launcher_liger.sh'
#   ssh gb10 'cd /home/dave/cppmega-bisect && git checkout -f 7a35918'
#   ssh gb10 'nohup bash /home/dave/gb10_bisect_launcher_liger.sh > /home/dave/logs/bisect_liger.log 2>&1 &'

set -euo pipefail

BISECT_ROOT="${BISECT_ROOT:-/home/dave/cppmega-bisect}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/home/dave/megatron-lm}"
VENV="${VENV:-/home/dave/cppmega-venv}"
LOG="${LOG:-/home/dave/logs/gb10_bisect_liger_$(date +%Y%m%d_%H%M%S).log}"

source "${VENV}/bin/activate"
export PYTHONPATH="${BISECT_ROOT}:${MEGATRON_ROOT}"

pip install --no-deps --force-reinstall -e "${BISECT_ROOT}" >/dev/null 2>&1

python -c "import cppmega; assert 'cppmega-bisect' in cppmega.__file__, cppmega.__file__; print('[import-ok]', cppmega.__file__)"

# 13-layer symmetric cut (matches bisect baseline).
export CPPMEGA_NEM_PATTERN="AEMEAEMEAEMR"
export CPPMEGA_LAYER_DEPTH=13
export CPPMEGA_DSA_A_LAYER_RANKS="1,2,3"
export CPPMEGA_NGRAM_HASH_ENABLED=0
export CPPMEGA_STRUCTURE_ENABLED=0
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRITON_CACHE_DIR=/home/dave/.triton-cache
export CPPMEGA_OPTIMIZER=muon
export CPPMEGA_MUON_SCALAR_OPTIMIZER=adam8bit
export CPPMEGA_MUON_QUANTIZED_MOMENTUM=1
export CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE=int8
export CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER=1
export CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER=1
export CPPMEGA_GRAD_REDUCE_IN_BF16=1
mkdir -p "${TRITON_CACHE_DIR}" /home/dave/logs

cd "${BISECT_ROOT}"

HYBRID_LAYER_PATTERN="$(python - <<'PY'
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern
print(build_default_hybrid_layer_pattern(mtp_depths=1))
PY
)"

NATIVE_ARGS="$(python - <<'PY'
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan
import os
plan = build_nam56r_feature_plan(
    pattern=os.environ["CPPMEGA_NEM_PATTERN"],
    depth=int(os.environ["CPPMEGA_LAYER_DEPTH"]),
    mtp_depths=1,
)
bundle = build_nam56r_megatron_native_args(
    plan=plan, enable_mla=True, enable_mtp=True,
    mtp_mode='hybrid', enable_moe=True,
)
print(bundle.to_shell_fragment())
PY
)"

# Wrapper dir + Liger-installing shim.  Mirrors bench3 cppmega_mimo_shim.py
# Liger-CE blocks (lines 422-435 of remote_smoke_h200_dsa_9_4_m.sh).
REMOTE_WORKDIR="$(mktemp -d /tmp/cppmega-bisect-liger-gb10.XXXXXX)"
trap 'rm -rf "${REMOTE_WORKDIR}"' EXIT
cp "${MEGATRON_ROOT}/pretrain_mamba.py" "${REMOTE_WORKDIR}/pretrain_mamba.py"
cat > "${REMOTE_WORKDIR}/mamba_builders.py" <<'PY'
from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder
PY
cat > "${REMOTE_WORKDIR}/model_provider.py" <<'PY'
from megatron.training import get_args
def model_provider(model_builder, pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    args = get_args()
    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)
PY

# THE shim: install Liger MTP + main-head CE.  Imported by pretrain_mamba.py
# entry via PYTHONPATH (REMOTE_WORKDIR is first).
cat > "${REMOTE_WORKDIR}/cppmega_liger_shim.py" <<'PY'
"""Install bench3-equivalent Liger CE patches on GB10 bisect runs."""
import sys

# (9) MTP Liger fused CE — mirrors bench3 cppmega_mimo_shim.py:425-426
from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
patch_mtp_loss_with_liger()
print("[gb10_liger_shim] MTP Liger CE installed", file=sys.stderr)

# (9b) Main-head linear CE fusion — mirrors :434-435
from cppmega.megatron.apply_linear_ce_patch import patch_mamba_output_layer_with_linear_ce
patch_mamba_output_layer_with_linear_ce()
print("[gb10_liger_shim] Main-head Liger CE installed", file=sys.stderr)
PY

# Inject `import cppmega_liger_shim` into pretrain_mamba.py prologue.
sed -i '1i import cppmega_liger_shim  # GB10 bisect: install Liger patches' \
  "${REMOTE_WORKDIR}/pretrain_mamba.py"

DATA_PATH="1.0 /home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10_train"

PYTHONPATH="${REMOTE_WORKDIR}:${BISECT_ROOT}:${MEGATRON_ROOT}" \
python -m torch.distributed.run --nproc_per_node=1 "${REMOTE_WORKDIR}/pretrain_mamba.py" \
  --data-path ${DATA_PATH} \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /home/dave/cppmega-root/cpp_tokenizer_hf \
  --vocab-size 65536 --make-vocab-size-divisible-by 128 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --no-gradient-accumulation-fusion --no-persist-layer-norm --no-masked-softmax-fusion \
  --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}" \
  --hidden-size 2048 --ffn-hidden-size 5632 --num-attention-heads 16 \
  --seq-length 2048 --max-position-embeddings 2048 \
  --micro-batch-size 8 --global-batch-size 8 --train-iters 5 \
  --eval-interval 50000000 --eval-iters 1 \
  --lr 1e-4 --min-lr 1e-5 --lr-decay-style constant \
  --optimizer "${CPPMEGA_OPTIMIZER}" \
  --muon-momentum 0.95 --muon-scale-mode spectral --muon-num-ns-steps 5 \
  --muon-tp-mode blockwise --muon-scalar-optimizer "${CPPMEGA_MUON_SCALAR_OPTIMIZER}" \
  --muon-quantized-momentum --muon-quantized-momentum-dtype "${CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE}" \
  --use-bf16-no-master-emerging-optimizer \
  --use-bf16-no-master-emerging-fallback-optimizer --grad-reduce-in-bf16 \
  --position-embedding-type rope --normalization RMSNorm \
  --disable-bias-linear --untie-embeddings-and-output-weights \
  --bf16 --fp8-format hybrid --fp8-recipe tensorwise \
  --fp8-amax-history-len 1024 --fp8-amax-compute-algo max \
  --use-mcore-models --transformer-impl transformer_engine --use-flash-attn --attention-backend flash \
  --spec cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec \
  --cross-entropy-loss-fusion --cross-entropy-fusion-impl linear \
  ${NATIVE_ARGS} --moe-token-dispatcher-type alltoall \
  --save-interval 50000000 --log-interval 1 \
  > "${LOG}" 2>&1 &

TRAIN_PID=$!
echo "[train] pid=${TRAIN_PID} log=${LOG}"
wait ${TRAIN_PID} || { echo "[train] exit=$?"; tail -n 150 "${LOG}"; exit 2; }
echo "=== TAIL ==="
tail -n 80 "${LOG}"
echo "=== GREP ==="
grep -E "grad norm|lm loss|gb10_liger_shim" "${LOG}" | head -30
