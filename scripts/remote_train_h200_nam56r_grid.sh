#!/bin/bash
# Parameterized NAM56R full 7/7 MIMO training launcher for bench3 grid search.
#
# Run ID: stream_A grid search 2026-04-12 / task #75
#
# All knobs below can be overridden via environment variables. Defaults match
# the 112k tok/sec baseline (PP=2 VPP=2 MBS=4 GBS=64 MTP=1 BF16 AllToAll MoE
# + per-module CUDA graphs).
#
# Hard constraints (see docs/nam56r_grid_search_2026_04_12.md):
#   * Real data only (clang_semantic_4k_v10_train)
#   * Full 7/7 Mamba3 MIMO features (nam56r_full_spec)
#   * TP=1 only
#   * cuDNN path fixed via ~/.bashrc
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/venv}"
RUN_ID="${RUN_ID:-nam56r_grid_run}"
LOG="${LOG:-${REMOTE_ROOT}/cppmega-root/cppmega/nam56r_grid/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-${REMOTE_ROOT}/cppmega-root/cppmega/nam56r_grid/${RUN_ID}_ckpt}"

mkdir -p "$(dirname "${LOG}")" "${CKPT_DIR}"
rm -rf "${CKPT_DIR}"/*  || true  # fresh init each run

# Activate venv
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega-root/cppmega:${REMOTE_ROOT}/cppmega-root/megatron-lm:${PYTHONPATH:-}"

# venv cuDNN must come first
_CL="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/cudnn/lib"
[ -d "${_CL}" ] && export LD_LIBRARY_PATH="${_CL}:${LD_LIBRARY_PATH:-}"

# NAM56R env defaults
export TILELANG_EXECUTION_BACKEND="${TILELANG_EXECUTION_BACKEND:-cython}"
export CPPMEGA_MAMBA3_MIMO="${CPPMEGA_MAMBA3_MIMO:-1}"
export CPPMEGA_MAMBA_NUM_GROUPS="${CPPMEGA_MAMBA_NUM_GROUPS:-8}"
export CPPMEGA_NEM_PATTERN="${CPPMEGA_NEM_PATTERN:-AEMEAEMEAEMR}"
export CPPMEGA_LAYER_DEPTH="${CPPMEGA_LAYER_DEPTH:-52}"
export CPPMEGA_R_LAYER_INDICES="${CPPMEGA_R_LAYER_INDICES:-12,24,36,48}"
export CPPMEGA_NGRAM_HASH_ENABLED="${CPPMEGA_NGRAM_HASH_ENABLED:-1}"
export CPPMEGA_NGRAM_HASH_ORDERS="${CPPMEGA_NGRAM_HASH_ORDERS:-2,3}"
export CPPMEGA_NGRAM_HASH_HEADS="${CPPMEGA_NGRAM_HASH_HEADS:-8}"
export CPPMEGA_NGRAM_HASH_TABLE_SIZE="${CPPMEGA_NGRAM_HASH_TABLE_SIZE:-500000}"
export CPPMEGA_NGRAM_HASH_EMBED_DIM="${CPPMEGA_NGRAM_HASH_EMBED_DIM:-16}"
export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-1}"
export CPPMEGA_STRUCTURE_COMPONENTS="${CPPMEGA_STRUCTURE_COMPONENTS:-core}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "${TRITON_CACHE_DIR}"

# Parallelism knobs
TP_SIZE="${TP_SIZE:-1}"             # MUST stay 1 — Mamba3 has no TP
PP_SIZE="${PP_SIZE:-2}"
VPP_SIZE="${VPP_SIZE:-2}"           # 0 = disable VPP
EP_SIZE="${EP_SIZE:-1}"
MBS="${MBS:-4}"
GBS="${GBS:-64}"
SEQ_LEN="${SEQ_LEN:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-30}"
MTP_DEPTHS="${MTP_DEPTHS:-1}"        # 0 disables MTP entirely
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-per_module}"  # off | per_module | full_iteration
FP8_MODE="${FP8_MODE:-off}"          # off | hybrid_mla_moe
NO_ROPE_FUSION="${NO_ROPE_FUSION:-1}" # required when PP>1 (fused MLA RoPE crashes)

# Build workdir with pretrain_mamba shim
WORKDIR=$(mktemp -d /tmp/cppmega-nam56r-grid.XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

# Copy the MIMO shim from exp0 if present — it installs the Mamba3 fp32 bias
# forward pre-hook which is required to keep Mamba3 biases in fp32 when wrapped
# in Float16Module.
cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Shim: install MIMO __post_init__ hook + Mamba3 fp32-bias forward pre-hook."""
from __future__ import annotations
import os
import sys

# (1) deprecate_inference_params compatibility shim
try:
    from megatron.core.inference.contexts import static_context as _sc
    if not hasattr(_sc, "deprecate_inference_params"):
        try:
            from megatron.core.utils import deprecate_inference_params as _dip
        except ImportError:
            def _dip(inference_context, inference_params):
                if inference_context is None and inference_params is not None:
                    return inference_params
                return inference_context
        _sc.deprecate_inference_params = _dip
except Exception as _exc:
    print(f"[cppmega_mimo_shim] static_context alias skipped: {_exc}", file=sys.stderr)

# (2) MIMO __post_init__
_mimo_on = os.environ.get("CPPMEGA_MAMBA3_MIMO", "0") == "1"
if _mimo_on:
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
        _orig_post_init = TransformerConfig.__post_init__
        def _cppmega_mimo_post_init(self):
            _orig_post_init(self)
            if not getattr(self, "cppmega_mamba3_is_mimo", False):
                object.__setattr__(self, "cppmega_mamba3_is_mimo", True)
            if not getattr(self, "cppmega_mamba3_mimo_rank", None):
                object.__setattr__(self, "cppmega_mamba3_mimo_rank", 4)
            if not getattr(self, "cppmega_mamba3_chunk_size", None):
                object.__setattr__(self, "cppmega_mamba3_chunk_size", 16)
        TransformerConfig.__post_init__ = _cppmega_mimo_post_init
        print("[cppmega_mimo_shim] MIMO patch installed (rank=4, chunk=16)")
    except Exception as _exc:
        print(f"[cppmega_mimo_shim] MIMO patch failed: {_exc}", file=sys.stderr)

# (3) Mamba3 fp32-bias forward pre-hook
try:
    from mamba_ssm.modules.mamba3 import Mamba3 as _Mamba3
    import torch as _torch
    _FP32_NAMES = ("B_bias", "C_bias", "D", "dt_bias", "mimo_x", "mimo_z", "mimo_o")
    def _restore_bias_fp32(module, _inputs):
        for _name in _FP32_NAMES:
            _p = getattr(module, _name, None)
            if _p is not None and _p.dtype != _torch.float32:
                _p.data = _p.data.float()
    if not getattr(_Mamba3, "_cppmega_fp32_bias_hook", False):
        _Mamba3._cppmega_fp32_bias_hook = True
        _orig_init = _Mamba3.__init__
        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.register_forward_pre_hook(_restore_bias_fp32)
        _Mamba3.__init__ = _patched_init
        print("[cppmega_mimo_shim] Mamba3 fp32-bias forward pre-hook installed")
except Exception as _exc:
    print(f"[cppmega_mimo_shim] Mamba3 fp32-bias hook failed: {_exc}", file=sys.stderr)

# (4) cppmega_mamba3_* __getattr__ fallback
try:
    from megatron.core.transformer.transformer_config import TransformerConfig
    _TC_BASE_GETATTR = getattr(TransformerConfig, "__getattr__", None)
    def _cppmega_getattr(self, name):
        if name.startswith("cppmega_mamba3_"):
            raise AttributeError(name)
        if _TC_BASE_GETATTR is not None:
            return _TC_BASE_GETATTR(self, name)
        raise AttributeError(name)
    if not hasattr(TransformerConfig, "_cppmega_mamba3_attr_patched"):
        TransformerConfig.__getattr__ = _cppmega_getattr
        TransformerConfig._cppmega_mamba3_attr_patched = True
except Exception:
    pass
PY

cp "${REMOTE_ROOT}/cppmega-root/megatron-lm/pretrain_mamba.py" "${WORKDIR}/pretrain_mamba_original.py"
cat > "${WORKDIR}/pretrain_mamba.py" <<'PY'
import os, sys, runpy
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
import cppmega_mimo_shim  # noqa: F401
runpy.run_path(
    os.path.join(_here, "pretrain_mamba_original.py"),
    run_name="__main__",
)
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

# Build hybrid layer pattern
#  - mtp_depths: 0 -> no /... suffix; >=1 -> suffix "*-" per depth
#  - VPP>1 -> split main into VPP*PP equal chunks separated by "|"
HYBRID_PATTERN=$(python - <<PY
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern

mtp_depths = ${MTP_DEPTHS}
pp = ${PP_SIZE}
vpp = ${VPP_SIZE}
p = build_default_hybrid_layer_pattern(mtp_depths=max(mtp_depths, 0))
if "/" in p:
    main, mtp_part = p.split("/", 1)
else:
    main, mtp_part = p, ""
# Drop mtp suffix entirely if mtp_depths==0
if mtp_depths == 0:
    mtp_part = ""
n_chunks = pp * max(vpp, 1)
if n_chunks > 1:
    total = len(main)
    per = total // n_chunks
    assert total % n_chunks == 0, f"cannot split {total}-layer main into {n_chunks} equal chunks"
    chunks = [main[i*per:(i+1)*per] for i in range(n_chunks)]
    main = "|".join(chunks)
print(main + (("/" + mtp_part) if mtp_part else ""))
PY
)
echo "HYBRID_PATTERN: ${HYBRID_PATTERN}"

# Native args (MLA + MoE + optional MTP, DSA stays off in grid search to reduce surface)
NATIVE_ARGS=$(python - <<PY
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan

mtp_depths = ${MTP_DEPTHS}
enable_mtp = mtp_depths > 0
plan = build_nam56r_feature_plan(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=max(mtp_depths, 1))
bundle = build_nam56r_megatron_native_args(
    plan=plan,
    enable_mla=True,
    enable_mtp=enable_mtp,
    mtp_mode="hybrid",
    mtp_num_predictors=mtp_depths,
    enable_moe=True,
    moe_expert_model_parallel_size=${EP_SIZE},
)
print(bundle.to_shell_fragment())
PY
)
echo "NATIVE_ARGS: ${NATIVE_ARGS}"


# VPP: no --num-layers-per-virtual-pipeline-stage flag when a pipe-separated
# --hybrid-layer-pattern is supplied. Megatron infers VPP count from the number
# of `|` segments in the pattern (see arguments.py:~line 550 assert). The chunk
# splitting already happened in the HYBRID_PATTERN python block above.
VPP_FLAG=""

# CUDA graph flags
CG_FLAGS=""
case "${CUDA_GRAPH_MODE}" in
  off)
    CG_FLAGS="--cuda-graph-impl none"
    ;;
  per_module)
    CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess --cuda-graph-warmup-steps 3"
    ;;
  per_module_moe)
    CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe --cuda-graph-warmup-steps 3"
    ;;
  attn_only)
    CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn --cuda-graph-warmup-steps 3"
    ;;
  full_iteration)
    CG_FLAGS="--cuda-graph-impl local --cuda-graph-scope full_iteration --cuda-graph-warmup-steps 3"
    ;;
  *)
    echo "ERROR: unknown CUDA_GRAPH_MODE=${CUDA_GRAPH_MODE}"; exit 2;;
esac

# FP8 flags (only MLA+MoE — never Mamba3 which has fp32 params)
FP8_FLAGS=""
if [ "${FP8_MODE}" = "hybrid_mla_moe" ]; then
  FP8_FLAGS="--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
fi

# No-rope-fusion needed if PP>1 (upstream bug)
ROPE_FLAG=""
if [ "${NO_ROPE_FUSION}" = "1" ]; then
  ROPE_FLAG="--no-rope-fusion"
fi

# MoE dispatcher + capacity args (required for CUDA graph compat)
MOE_EXTRA_FLAGS="--moe-token-dispatcher-type alltoall"
if [ "${CUDA_GRAPH_MODE}" != "off" ]; then
  MOE_EXTRA_FLAGS="${MOE_EXTRA_FLAGS} --moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0 --moe-permute-fusion"
fi

# Pre-flight import check
python -c 'import cppmega, megatron; print("cppmega", cppmega.__version__)'

# Guard: refuse to launch with NullTokenizer or mock-data flags. Only the
# python -m torch.distributed.run invocation is inspected. Since the flags live
# in this shell script at known positions we just grep the rendered command.
if grep -E "^\s*--(tokenizer-type NullTokenizer|mock-data)\b" "$0" > /dev/null 2>&1; then
  echo "ERROR: this launcher contains forbidden NullTokenizer/mock-data tokens"
  exit 9
fi

echo "=== NAM56R full 7/7 MIMO grid run ==="
echo "RUN_ID=${RUN_ID} TP=${TP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE} EP=${EP_SIZE} MBS=${MBS} GBS=${GBS} MTP=${MTP_DEPTHS} CG=${CUDA_GRAPH_MODE} FP8=${FP8_MODE}"
echo "LOG=${LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

python -m torch.distributed.run --nproc_per_node=8 "${WORKDIR}/pretrain_mamba.py" \
  --data-path 1.0 "${REMOTE_ROOT}/data/megatron/clang_semantic_4k_v10_train" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${REMOTE_ROOT}/tokenizer" \
  --split 98,1,1 \
  --tensor-model-parallel-size ${TP_SIZE} \
  --pipeline-model-parallel-size ${PP_SIZE} \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  ${ROPE_FLAG} \
  --hybrid-layer-pattern "${HYBRID_PATTERN}" \
  --hidden-size 3584 \
  --ffn-hidden-size 18944 \
  --num-attention-heads 28 \
  --seq-length ${SEQ_LEN} \
  --max-position-embeddings ${SEQ_LEN} \
  --micro-batch-size ${MBS} \
  --global-batch-size ${GBS} \
  --train-iters ${TRAIN_ITERS} \
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
  --use-flash-attn \
  --attention-backend flash \
  --spec cppmega.megatron.nam56r_full_spec build_cppmega_nam56r_full_stack_spec \
  ${VPP_FLAG} \
  ${CG_FLAGS} \
  ${FP8_FLAGS} \
  ${MOE_EXTRA_FLAGS} \
  --no-check-for-nan-in-loss-and-grad \
  ${NATIVE_ARGS} \
  --save "${CKPT_DIR}" \
  --load "${CKPT_DIR}" \
  --save-interval 1000000 \
  --log-interval 1 \
  > "${LOG}" 2>&1

EXIT_CODE=$?
echo "=== Exit code: ${EXIT_CODE} ==="
echo "=== Last 60 lines of ${LOG} ==="
tail -60 "${LOG}"
exit ${EXIT_CODE}
