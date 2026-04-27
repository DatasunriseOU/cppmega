#!/usr/bin/env bash
# ============================================================================
# DEPRECATED / HISTORICAL (2026-04-14)
# ----------------------------------------------------------------------------
# Original purpose (2026-04-12): sequential sweep of 8 PP=2 configs on bench3
# to find the best DSA 9+4 topology. That exploration has CONCLUDED:
# PP=1 EP=8 MBS=10 FP8 tensorwise (v3) is the bench3 production config at
# 268 TFLOP/s, and PP=1 EP=4 MBS=8 BF16 is europe at 289 TFLOP/s. See
# `docs/production_status.md` for the canonical table.
#
# ALL B1-B8 configs below use PP=2 VPP=2 MBS=4/8 EP=1 — this is the old
# "Stream L" topology which measures ~193 TFLOP/s on europe (12-month-old
# baseline). They do NOT reproduce the current 268/289 production records.
#
# Additionally, the FP8 indexer path this script originally invoked
# (`cppmega.megatron.dsa_fp8_patch`) has been REMOVED upstream. The shim
# below has been rewritten to use `dsa_indexer_fused_patch` (the current
# BF16 per-head fused accumulator — memory-equivalent, no FP8 amax cost,
# production-ready). The CPPMEGA_DSA_INDEXER_DTYPE=fp8 env var is now a
# no-op for this script.
#
# Use this script ONLY for historical PP=2 config comparisons. For current
# production runs use `scripts/remote_smoke_h200_dsa_9_4_m.sh` with
# VARIANT=v3 (bench3) or VARIANT=v1 (europe). See `docs/production_status.md`.
# ============================================================================
# NAM56R DSA 9+4 PP=2 sweep (2026-04-12, historical)
#
# Sequential sweep of 8 PP=2 configs on bench3 (h200_1),
# 50 iters each. Measures tok/sec at iter 30-50 steady state.
#
# All configs share:
#   TP=1, DSA 9+4 (CPPMEGA_DSA_A_LAYER_RANKS="1,2,3,5,6,7,9,10,11"),
#   BF16 fused indexer (via dsa_indexer_fused_patch; FP8 path deleted),
#   sparse attention (CPPMEGA_DSA_SPARSE_MODE=tilelang, fallback gather_scatter),
#   real HF tokenizer + clang_semantic data, --enable-dsa,
#   --dsa-indexer-topk 256, --dsa-indexer-loss-coeff 0.001
#
# Usage (on bench3):
#   CONFIG=B1 bash scripts/remote_sweep_h200_dsa_production.sh
#   CONFIG=B2 bash scripts/remote_sweep_h200_dsa_production.sh
#   ...
#
# Or run the driver script to run all 8 sequentially:
#   bash scripts/drive_sweep_dsa_production.sh
set -euo pipefail

CPPMEGA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CPPMEGA_SCRIPT_DIR}/lib/deprecated_guard.sh"
cppmega_deprecated_script_guard "$(basename "$0")" \
  "scripts/remote_smoke_h200_dsa_9_4_m.sh with VARIANT=v3 or VARIANT=v1"

# Source .bashrc for cuDNN/NCCL paths
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"

CONFIG="${CONFIG:-B1}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/venv}"
RUN_ID="${RUN_ID:-nam56r_dsa_sweep_${CONFIG}}"
LOG_DIR="${REMOTE_ROOT}/cppmega-root/cppmega/nam56r_dsa_sweep"
LOG="${LOG:-${LOG_DIR}/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-${LOG_DIR}/${RUN_ID}_ckpt}"
RESULTS_FILE="${LOG_DIR}/sweep_results.jsonl"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
rm -rf "${CKPT_DIR}"/* || true

# Activate venv
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega-root/cppmega:${REMOTE_ROOT}/cppmega-root/megatron-lm:${PYTHONPATH:-}"

# venv cuDNN must come first
_NV="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia"
_LD_PREFIX=""
for _pkg in cudnn nccl cublas; do
  _d="${_NV}/${_pkg}/lib"
  [ -d "${_d}" ] && _LD_PREFIX="${_d}:${_LD_PREFIX}"
done
if [ -n "${_LD_PREFIX}" ]; then
  export LD_LIBRARY_PATH="${_LD_PREFIX}${LD_LIBRARY_PATH:-}"
fi

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

# DSA 9+4 permanent layout
export CPPMEGA_DSA_A_LAYER_RANKS="${CPPMEGA_DSA_A_LAYER_RANKS:-1,2,3,5,6,7,9,10,11}"
# DSA indexer dtype env var (legacy, no-op after dsa_fp8_patch removal;
# fused BF16 per-head accumulator is now the only path).
export CPPMEGA_DSA_INDEXER_DTYPE="${CPPMEGA_DSA_INDEXER_DTYPE:-bf16}"
# Sparse attention mode (tilelang fused kernel, fallback to gather_scatter)
export CPPMEGA_DSA_SPARSE_MODE="${CPPMEGA_DSA_SPARSE_MODE:-tilelang}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "${TRITON_CACHE_DIR}"

# =====================================================================
# Config definitions: PP VPP MBS GBS MTP_DEPTHS RECOMPUTE CG EP
# =====================================================================
TP_SIZE=1
SEQ_LEN=4096
TRAIN_ITERS=50
NO_ROPE_FUSION=1
DSA_TOPK=256
DSA_LOSS_COEFF=0.001

case "${CONFIG}" in
  B1)
    PP_SIZE=2; VPP_SIZE=2; MBS=4; GBS=64; MTP_DEPTHS=2
    RECOMPUTE_MODE="moe_act"
    CUDA_GRAPH_MODE="per_module"
    EP_SIZE=1
    ;;
  B2)
    PP_SIZE=2; VPP_SIZE=2; MBS=4; GBS=64; MTP_DEPTHS=2
    RECOMPUTE_MODE="moe_act"
    CUDA_GRAPH_MODE="off"
    EP_SIZE=1
    ;;
  B3)
    PP_SIZE=2; VPP_SIZE=2; MBS=8; GBS=64; MTP_DEPTHS=2
    RECOMPUTE_MODE="moe_act"
    CUDA_GRAPH_MODE="per_module"
    EP_SIZE=1
    ;;
  B4)
    PP_SIZE=2; VPP_SIZE=2; MBS=2; GBS=64; MTP_DEPTHS=2
    RECOMPUTE_MODE="moe_act"
    CUDA_GRAPH_MODE="per_module"
    EP_SIZE=1
    ;;
  B5)
    PP_SIZE=2; VPP_SIZE=2; MBS=4; GBS=128; MTP_DEPTHS=2
    RECOMPUTE_MODE="moe_act"
    CUDA_GRAPH_MODE="per_module"
    EP_SIZE=1
    ;;
  B6)
    PP_SIZE=2; VPP_SIZE=2; MBS=4; GBS=64; MTP_DEPTHS=1
    RECOMPUTE_MODE="moe_act"
    CUDA_GRAPH_MODE="per_module"
    EP_SIZE=1
    ;;
  B7)
    PP_SIZE=2; VPP_SIZE=2; MBS=4; GBS=64; MTP_DEPTHS=2
    RECOMPUTE_MODE="full_selective"
    CUDA_GRAPH_MODE="per_module"
    EP_SIZE=1
    ;;
  B8)
    PP_SIZE=2; VPP_SIZE=1; MBS=4; GBS=64; MTP_DEPTHS=2
    RECOMPUTE_MODE="moe_act"
    CUDA_GRAPH_MODE="per_module"
    EP_SIZE=1
    ;;
  *)
    echo "ERROR: unknown CONFIG=${CONFIG} (expected B1..B8)" >&2
    exit 2
    ;;
esac

# Build workdir with pretrain_mamba shim
WORKDIR=$(mktemp -d /tmp/cppmega-nam56r-dsa-sweep.XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Shim: MIMO + fp32-bias + DSA fused indexer + sparse attention + peak-memory reporter."""
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

# (5) DSA indexer fused patch (BF16 per-head bmm, current path)
# Historical note: CPPMEGA_DSA_INDEXER_DTYPE=fp8 used to invoke
# cppmega.megatron.dsa_fp8_patch.apply_dsa_fp8_patch — that module has
# been removed. The fused BF16 indexer is memory-equivalent and the
# only supported path. The env var is preserved here only to keep
# legacy invocation surface compatible; any value other than "bf16"
# is ignored (with a warning).
_dsa_dtype = os.environ.get("CPPMEGA_DSA_INDEXER_DTYPE", "bf16").lower()
print(f"[cppmega_mimo_shim] CPPMEGA_DSA_INDEXER_DTYPE='{_dsa_dtype}' (fused BF16 path is the only supported path)")
if _dsa_dtype != "bf16":
    print(f"[cppmega_mimo_shim] WARNING: CPPMEGA_DSA_INDEXER_DTYPE='{_dsa_dtype}' is legacy; FP8 patch removed, using fused BF16", file=sys.stderr)
try:
    from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
    _applied = apply_dsa_indexer_fused_patch()
    print(f"[cppmega_mimo_shim] DSA indexer fused patch applied={_applied}")
except Exception as _exc:
    print(f"[cppmega_mimo_shim] DSA indexer fused patch failed: {_exc}", file=sys.stderr)
    raise

_sparse_mode = os.environ.get("CPPMEGA_DSA_SPARSE_MODE", "tilelang").lower()
print(f"[cppmega_mimo_shim] CPPMEGA_DSA_SPARSE_MODE resolves to '{_sparse_mode}'")

# (6) Per-rank peak-memory reporter (atexit hook)
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
            f"[sweep_peak_mem] rank={rank} device={dev} "
            f"peak_alloc_gib={peak_alloc:.3f} peak_reserved_gib={peak_reserved:.3f}",
            flush=True,
        )
    except Exception as _exc:
        print(f"[sweep_peak_mem] report failed: {_exc}", file=sys.stderr)

atexit.register(_cppmega_peak_mem_report)
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

# Native args (MLA + MoE + MTP + DSA) with custom topk and loss_coeff
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
    enable_dsa=True,
    dsa_indexer_dtype="bf16",
    dsa_indexer_topk=${DSA_TOPK},
    dsa_indexer_loss_coeff=${DSA_LOSS_COEFF},
)
print(bundle.to_shell_fragment())
PY
)
echo "NATIVE_ARGS (pre-sed): ${NATIVE_ARGS}"



# EP override
echo "NATIVE_ARGS (post-sed): ${NATIVE_ARGS}"

# CUDA graph flags
CG_FLAGS=""
case "${CUDA_GRAPH_MODE}" in
  off)
    CG_FLAGS="--cuda-graph-impl none"
    ;;
  per_module)
    CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess --cuda-graph-warmup-steps 3"
    ;;
  *)
    echo "ERROR: unknown CUDA_GRAPH_MODE=${CUDA_GRAPH_MODE}"; exit 2;;
esac

# MoE flags (required for CUDA graph compat)
MOE_EXTRA_FLAGS="--moe-token-dispatcher-type alltoall"
if [ "${CUDA_GRAPH_MODE}" != "off" ]; then
  MOE_EXTRA_FLAGS="${MOE_EXTRA_FLAGS} --moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0 --moe-permute-fusion"
fi

# Recompute flags
RECOMPUTE_FLAGS=""
case "${RECOMPUTE_MODE}" in
  moe_act)
    RECOMPUTE_FLAGS="--recompute-granularity selective --recompute-modules moe_act"
    ;;
  full_selective)
    RECOMPUTE_FLAGS="--recompute-granularity selective"
    ;;
  *)
    echo "ERROR: unknown RECOMPUTE_MODE=${RECOMPUTE_MODE}"; exit 2;;
esac

# RoPE fusion flag
ROPE_FLAG=""
if [ "${NO_ROPE_FUSION}" = "1" ]; then
  ROPE_FLAG="--no-rope-fusion"
fi

# Pre-flight checks
python -c 'import cppmega, megatron; print("cppmega", cppmega.__version__)'
python -c "from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch; print('dsa_indexer_fused_patch importable')"

# Validate DSA A-layer rank parsing
python - <<PY
import os
os.environ.setdefault("CPPMEGA_DSA_A_LAYER_RANKS", "${CPPMEGA_DSA_A_LAYER_RANKS}")
from cppmega.megatron.nam56r_layout import load_attention_layer_numbers, load_dsa_a_layer_ranks
attn_nums = load_attention_layer_numbers()
dsa_ranks = load_dsa_a_layer_ranks()
mla_ranks = [r for r in range(len(attn_nums)) if r not in dsa_ranks]
print(f"[DSA-9-4] A-layer layer_numbers (1-indexed): {list(attn_nums)}")
print(f"[DSA-9-4] DSA A-ranks: {list(dsa_ranks)}")
print(f"[DSA-9-4] DSA layer_numbers: {[attn_nums[r] for r in dsa_ranks]}")
print(f"[DSA-9-4] MLA A-ranks: {mla_ranks}")
print(f"[DSA-9-4] MLA layer_numbers: {[attn_nums[r] for r in mla_ranks]}")
assert len(dsa_ranks) == 9 and len(mla_ranks) == 4, (
    f"expected 9 DSA + 4 MLA, got {len(dsa_ranks)} + {len(mla_ranks)}"
)
PY

# Guard: refuse NullTokenizer or mock-data
_FORBID1="Null""Tokenizer"
_FORBID2="--mock""-data"
if echo "${NATIVE_ARGS} ${CG_FLAGS} ${MOE_EXTRA_FLAGS} ${ROPE_FLAG} ${RECOMPUTE_FLAGS}" | grep -E "(${_FORBID1}|${_FORBID2})" > /dev/null; then
  echo "ERROR: rendered command contains ${_FORBID1} or ${_FORBID2}"
  exit 9
fi

echo "=== NAM56R DSA 9+4 production sweep: ${CONFIG} ==="
echo "RUN_ID=${RUN_ID} CONFIG=${CONFIG}"
echo "TP=${TP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE} EP=${EP_SIZE} MBS=${MBS} GBS=${GBS} MTP=${MTP_DEPTHS}"
echo "RECOMPUTE=${RECOMPUTE_MODE} CG=${CUDA_GRAPH_MODE}"
echo "DSA_TOPK=${DSA_TOPK} DSA_LOSS_COEFF=${DSA_LOSS_COEFF}"
echo "DSA_SPARSE_MODE=${CPPMEGA_DSA_SPARSE_MODE}"
echo "DSA_A_LAYER_RANKS=${CPPMEGA_DSA_A_LAYER_RANKS}"
echo "LOG=${LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

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
  ${CG_FLAGS} \
  ${MOE_EXTRA_FLAGS} \
  --no-check-for-nan-in-loss-and-grad \
  ${NATIVE_ARGS} \
  ${RECOMPUTE_FLAGS} \
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
exit ${EXIT_CODE}
