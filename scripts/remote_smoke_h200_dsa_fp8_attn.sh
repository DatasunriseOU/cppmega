#!/usr/bin/env bash
# FP8 sparse attention test: NAM56R DSA 9+4 with FP8 attention on top of the surviving bf16 indexer path.
#
# Tests the FP8 TileLang sparse MLA forward kernel integration:
#   - CPPMEGA_DSA_FP8_ATTENTION=1 enables FP8 forward + BF16 backward
#   - PP=1 (single pipeline stage for cleaner profiling)
#   - MBS=4, 10 iters
#   - Compare TFLOP/s with BF16 baseline (~237) -- expect ~280-350
#   - Check grad_norm stability
#
# Run on europe (h200_2):
#   VARIANT=fp8 bash scripts/remote_smoke_h200_dsa_fp8_attn.sh
#
# Compare with BF16 baseline:
#   VARIANT=bf16 bash scripts/remote_smoke_h200_dsa_fp8_attn.sh
set -euo pipefail

[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"

REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
VARIANT="${VARIANT:-fp8}"
RUN_ID="${RUN_ID:-cppmega_nam56r_dsa_fp8_attn_${VARIANT}}"
LOG="${LOG:-${REMOTE_ROOT}/cppmega/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-${REMOTE_ROOT}/cppmega/nam56r_grid/${RUN_ID}_ckpt}"

mkdir -p "$(dirname "${LOG}")" "${CKPT_DIR}"
rm -rf "${CKPT_DIR}"/* || true

source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"

_NV="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia"
_LD_PREFIX=""
for _pkg in cudnn nccl cublas; do
  _d="${_NV}/${_pkg}/lib"
  [ -d "${_d}" ] && _LD_PREFIX="${_d}:${_LD_PREFIX}"
done
export LD_LIBRARY_PATH="${_LD_PREFIX}/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"

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
export CPPMEGA_DSA_A_LAYER_RANKS="${CPPMEGA_DSA_A_LAYER_RANKS:-1,2,3,5,6,7,9,10,11}"
# Live DSA indexer path is bf16-only; FP8 indexer path was removed 2026-04-13.
export CPPMEGA_DSA_INDEXER_DTYPE="${CPPMEGA_DSA_INDEXER_DTYPE:-bf16}"
# Use gather_scatter for DSA sparse attention baseline
export CPPMEGA_DSA_SPARSE_MODE="${CPPMEGA_DSA_SPARSE_MODE:-gather_scatter}"

# FP8 attention: variant switch
case "${VARIANT}" in
  fp8)
    echo "[fp8_attn] FP8 sparse attention forward enabled"
    export CPPMEGA_DSA_FP8_ATTENTION=1
    ;;
  bf16)
    echo "[fp8_attn] BF16 baseline (no FP8 attention)"
    export CPPMEGA_DSA_FP8_ATTENTION=0
    ;;
  *)
    echo "ERROR: unknown VARIANT=${VARIANT} (expected fp8|bf16)" >&2
    exit 2
    ;;
esac

export CPPMEGA_MAMBA_RECOMPUTE="${CPPMEGA_MAMBA_RECOMPUTE:-1}"
export CPPMEGA_DSA_SKIP_INDEXER_LOSS="${CPPMEGA_DSA_SKIP_INDEXER_LOSS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
unset NCCL_NET NCCL_NET_PLUGIN 2>/dev/null || true
export NCCL_NET_PLUGIN=none
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "${TRITON_CACHE_DIR}"

# PP=1 for cleaner profiling, EP=1 (all experts on each rank).
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
VPP_SIZE="${VPP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
MBS="${MBS:-4}"
GBS="${GBS:-32}"
SEQ_LEN="${SEQ_LEN:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
MTP_DEPTHS="${MTP_DEPTHS:-2}"
NO_ROPE_FUSION="${NO_ROPE_FUSION:-1}"

# Build workdir with pretrain_mamba shim
WORKDIR=$(mktemp -d /tmp/cppmega-nam56r-dsa-fp8-attn.XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Shim: install MIMO __post_init__ hook + Mamba3 fp32-bias forward pre-hook
+ Stream E/G DSA FP8 fwd+bwd patch + FP8 attention patch + loss_coeff==0 gate
+ per-rank peak-memory reporter."""
from __future__ import annotations
import os
import sys
import atexit

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

# (5) DSA indexer path (bf16-only after FP8 indexer removal) + loss_coeff==0 gate + FP8 attention
_dsa_dtype = os.environ.get("CPPMEGA_DSA_INDEXER_DTYPE", "bf16").lower()
_fp8_attn = os.environ.get("CPPMEGA_DSA_FP8_ATTENTION", "0")
print(f"[cppmega_mimo_shim] CPPMEGA_DSA_INDEXER_DTYPE={_dsa_dtype}")
print(f"[cppmega_mimo_shim] CPPMEGA_DSA_FP8_ATTENTION={_fp8_attn}")
if _dsa_dtype == "fp8":
    raise RuntimeError(
        "CPPMEGA_DSA_INDEXER_DTYPE=fp8 is no longer supported: dsa_fp8_patch.py "
        "and dsa_fp8_indexer.py were deleted on 2026-04-13. Use bf16."
    )

# (6) Mamba/M2RNN activation checkpointing
_mamba_recompute = os.environ.get("CPPMEGA_MAMBA_RECOMPUTE", "0") == "1"
if _mamba_recompute:
    try:
        from cppmega.megatron.mamba_recompute_patch import apply_mamba_recompute_patch
        _applied = apply_mamba_recompute_patch()
        print(f"[cppmega_mimo_shim] Mamba recompute patch applied={_applied}")
    except Exception as _exc:
        print(f"[cppmega_mimo_shim] Mamba recompute patch failed: {_exc}", file=sys.stderr)
        raise

# (7) Per-rank peak-memory reporter
def _cppmega_peak_mem_report():
    try:
        import torch
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        peak_alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(dev) / (1024 ** 3)
        rank = int(os.environ.get("RANK", "0"))
        print(
            f"[fp8_attn_peak_mem] rank={rank} device={dev} "
            f"peak_alloc_gib={peak_alloc:.3f} peak_reserved_gib={peak_reserved:.3f}",
            flush=True,
        )
    except Exception as _exc:
        print(f"[fp8_attn_peak_mem] report failed: {_exc}", file=sys.stderr)

atexit.register(_cppmega_peak_mem_report)
PY

cp "${REMOTE_ROOT}/megatron-lm/pretrain_mamba.py" "${WORKDIR}/pretrain_mamba_original.py"
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

# Build hybrid layer pattern -- PP=1 VPP=1, single chunk
HYBRID_LAYER_PATTERN=$(python - <<PY
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern

mtp_depths = ${MTP_DEPTHS}
pp = ${PP_SIZE}
vpp = ${VPP_SIZE}
p = build_default_hybrid_layer_pattern(mtp_depths=max(mtp_depths, 0))
if "/" in p:
    main, mtp_part = p.split("/", 1)
else:
    main, mtp_part = p, ""
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
echo "HYBRID_LAYER_PATTERN: ${HYBRID_LAYER_PATTERN}"

# Native args (MLA + MoE + MTP + DSA)
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
    moe_token_dispatcher_type="alltoall",
    moe_router_dtype=None,
    enable_dsa=True,
)
print(bundle.to_shell_fragment())
PY
)
echo "NATIVE_ARGS (pre-sed): ${NATIVE_ARGS}"

# EP=1: render alltoall directly through the helper.
MOE_EXTRA_FLAGS="--moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0"

echo "NATIVE_ARGS (post-sed): ${NATIVE_ARGS}"

EXTRA_FLAGS="--recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj --mla-down-proj-fusion --clip-grad 1.0"

ROPE_FLAG=""
if [ "${NO_ROPE_FUSION}" = "1" ]; then
  ROPE_FLAG="--no-rope-fusion"
fi

python -c 'import cppmega, megatron; print("cppmega", cppmega.__version__)'
python - <<PY
import os
dtype = os.environ.get("CPPMEGA_DSA_INDEXER_DTYPE", "bf16").lower()
assert dtype == "bf16", f"expected bf16 live DSA indexer path, got {dtype!r}"
print("live DSA indexer path validated: bf16 only")
print(f"fp8 attention env={os.environ.get('CPPMEGA_DSA_FP8_ATTENTION', '0')}")
PY
python -c "from cppmega.megatron.sparse_mla_ops.sparse_mla import SparseMLA_FP8, fused_sparse_mla_absorbed_fp8; print('SparseMLA_FP8 importable')"

echo "=== FP8 Attention Test: ${VARIANT} ==="
echo "RUN_ID=${RUN_ID} VARIANT=${VARIANT} TP=${TP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE} EP=${EP_SIZE} MBS=${MBS} GBS=${GBS} MTP=${MTP_DEPTHS}"
echo "CPPMEGA_DSA_FP8_ATTENTION=${CPPMEGA_DSA_FP8_ATTENTION:-0}"
echo "CPPMEGA_DSA_INDEXER_DTYPE=${CPPMEGA_DSA_INDEXER_DTYPE}"
echo "LOG=${LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

python -m torch.distributed.run --nproc_per_node=8 "${WORKDIR}/pretrain_mamba.py" \
  --data-path 1.0 "${REMOTE_ROOT}/data/megatron/clang_semantic_4k_v10_train" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${REMOTE_ROOT}/data/tokenizer" \
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
  --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}" \
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
  --lr 1e-5 \
  --min-lr 1e-6 \
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
  ${MOE_EXTRA_FLAGS} \
  --no-check-for-nan-in-loss-and-grad \
  ${NATIVE_ARGS} \
  ${EXTRA_FLAGS} \
  --save "${CKPT_DIR}" \
  --load "${CKPT_DIR}" \
  --save-interval 1000000 \
  --log-interval 1 \
  --log-throughput \
  > "${LOG}" 2>&1

EXIT_CODE=$?
echo "=== Exit code: ${EXIT_CODE} ==="
echo "=== Last 80 lines of ${LOG} ==="
tail -80 "${LOG}"

# Print throughput and grad_norm summary for comparison
echo ""
echo "=== Throughput and grad_norm summary ==="
grep -E "(throughput|TFLOP|grad_norm|tokens-per-sec)" "${LOG}" | tail -20
echo ""
echo "=== Peak memory ==="
grep -E "peak_(alloc|reserved)" "${LOG}" | tail -10

exit ${EXIT_CODE}
