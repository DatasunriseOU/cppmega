#!/usr/bin/env bash
# Stream L (task #89): NAM56R DSA 9+4 + FP8 indexer expert-parallel sweep.
#
# Spreads the 16-expert MoE across ranks via --expert-model-parallel-size to
# free stage-0/1 activation memory on top of the DSA 9+4 + FP8 indexer config.
# Topology is locked to the 112k baseline: TP=1 PP=2 VPP=2 MBS=4 GBS=64 MTP=2.
#
# VARIANT=v1 -> EP=2 DP=2 (8 experts per rank, 2x data-parallel)
# VARIANT=v2 -> EP=4 DP=1 (4 experts per rank, gradient accumulation handles GBS)
#
# Stream D v2 + Stream J revealed that dsa.py:202 compute_dsa_indexer_loss
# allocates ~7.5 GiB FP32 [b*np, sq, sk] per DSA layer regardless of
# loss_coeff=0.0. Stream L tests whether EP-based activation savings can
# offset that fixed ceiling enough to fit PP=2 stage-0.
#
# NO feature disable hacks. NO MBS drop. NO DSA off. NO MTP off. NO MoE off.
# NO CppmegaMamba3TPMixer. Real data only. Mamba-3 MIMO 7/7 preserved.
#
# Designed to run directly on LOCATION_2 (h200_1)
# inside a tmux session launched with `bash -l`. No gcloud/scp wrapping here
# -- this script IS the remote body.
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
VARIANT="${VARIANT:-v1}"
RUN_ID="${RUN_ID:-cppmega_nam56r_dsa_9_4_fp8_l_${VARIANT}}"
LOG="${LOG:-${REMOTE_ROOT}/cppmega/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-${REMOTE_ROOT}/cppmega/nam56r_grid/${RUN_ID}_ckpt}"

mkdir -p "$(dirname "${LOG}")" "${CKPT_DIR}"
rm -rf "${CKPT_DIR}"/* || true  # fresh init each run

# Activate venv
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"

# venv cuDNN must come first (MLA engine requires bundled cuDNN)
_CL="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/cudnn/lib"
[ -d "${_CL}" ] && export LD_LIBRARY_PATH="${_CL}:${LD_LIBRARY_PATH:-}"

# NAM56R env defaults (same as Stream D v2 / Stream J)
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
# Stream E+G FP8 DSA indexer (fwd + bwd patched)
export CPPMEGA_DSA_INDEXER_DTYPE="${CPPMEGA_DSA_INDEXER_DTYPE:-fp8}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
unset NCCL_NET NCCL_NET_PLUGIN 2>/dev/null || true
export NCCL_NET_PLUGIN=none
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "${TRITON_CACHE_DIR}"

# Parallelism defaults: 112k baseline topology (PP=2 VPP=2 MBS=4 GBS=64 MTP=2).
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-2}"
VPP_SIZE="${VPP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-1}"   # overridden per variant below
MBS="${MBS:-4}"
GBS="${GBS:-64}"
SEQ_LEN="${SEQ_LEN:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-120}"
MTP_DEPTHS="${MTP_DEPTHS:-2}"
NO_ROPE_FUSION="${NO_ROPE_FUSION:-1}"

# ---------------------------------------------------------------------------
# Variant overrides (v1 = EP=2 DP=2, v2 = EP=4 DP=1)
# ---------------------------------------------------------------------------
case "${VARIANT}" in
  v1)
    echo "[stream_l] V1 = EP=2 DP=2 PP=2 VPP=2 (8 experts/rank, 2x DP)"
    EP_SIZE=2
    ;;
  v2)
    echo "[stream_l] V2 = EP=4 DP=1 PP=2 VPP=2 (4 experts/rank, grad-accum GBS)"
    EP_SIZE=4
    ;;
  *)
    echo "ERROR: unknown VARIANT=${VARIANT} (expected v1|v2)" >&2
    exit 2
    ;;
esac

# Build workdir with pretrain_mamba shim (MIMO __post_init__ + fp32-bias hook
# + DSA FP8 patch + peak-memory reporter). Identical body to Stream J so
# downstream comparison is apples-to-apples.
WORKDIR=$(mktemp -d /tmp/cppmega-nam56r-dsa-9-4-l.XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Shim: install MIMO __post_init__ hook + Mamba3 fp32-bias forward pre-hook
+ Stream E/G DSA FP8 fwd+bwd patch + per-rank peak-memory reporter."""
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

# (5) Stream E+G: DSA FP8 fwd+bwd indexer patch — REMOVED 2026-04-13
# dsa_fp8_patch.py / dsa_fp8_indexer.py were deleted. Single DSA path is
# lemyx + IndexCache; FP8 indexer is no longer a supported variant.
_dsa_dtype = os.environ.get("CPPMEGA_DSA_INDEXER_DTYPE", "bf16").lower()
print(f"[cppmega_mimo_shim] CPPMEGA_DSA_INDEXER_DTYPE resolves to '{_dsa_dtype}' (fp8 path removed)")
if _dsa_dtype == "fp8":
    raise RuntimeError(
        "CPPMEGA_DSA_INDEXER_DTYPE=fp8 is no longer supported: dsa_fp8_patch.py "
        "and dsa_fp8_indexer.py were deleted on 2026-04-13. Use bf16."
    )

# (6) Stream L: per-rank peak-memory reporter (atexit hook)
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
            f"[stream_l_peak_mem] rank={rank} device={dev} "
            f"peak_alloc_gib={peak_alloc:.3f} peak_reserved_gib={peak_reserved:.3f}",
            flush=True,
        )
    except Exception as _exc:
        print(f"[stream_l_peak_mem] report failed: {_exc}", file=sys.stderr)

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

# Build hybrid layer pattern -- PP*VPP equal chunks separated by "|".
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

# Native args (MLA + MoE + MTP + DSA).
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
)
print(bundle.to_shell_fragment())
PY
)
echo "NATIVE_ARGS (pre-sed): ${NATIVE_ARGS}"


# Override --expert-model-parallel-size to EP_SIZE (always runs for Stream L:
# both variants set EP_SIZE >=2).


# Confirm the substitution actually happened.
if ! echo "${NATIVE_ARGS}" | grep -q -- "--expert-model-parallel-size ${EP_SIZE}"; then
  echo "ERROR: failed to set --expert-model-parallel-size ${EP_SIZE}; NATIVE_ARGS=${NATIVE_ARGS}" >&2
  exit 3
fi
echo "NATIVE_ARGS (post-sed): ${NATIVE_ARGS}"

# Per-module CUDA graph scope (matches 112k baseline).
CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess --cuda-graph-warmup-steps 3"
MOE_EXTRA_FLAGS="--moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0"

ROPE_FLAG=""
if [ "${NO_ROPE_FUSION}" = "1" ]; then
  ROPE_FLAG="--no-rope-fusion"
fi

python -c 'import cppmega, megatron; print("cppmega", cppmega.__version__)'
# dsa_fp8_patch / dsa_fp8_indexer import-checks removed 2026-04-13 — those
# modules were deleted; single DSA path is lemyx + IndexCache.

# Validate DSA A-layer rank parsing.
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

# Guard: refuse to launch if forbidden tokenizer/mock-data flags are present.
_FORBID1="Null""Tokenizer"
_FORBID2="--mock""-data"
if echo "${NATIVE_ARGS} ${CG_FLAGS} ${MOE_EXTRA_FLAGS} ${ROPE_FLAG}" | grep -E "(${_FORBID1}|${_FORBID2})" > /dev/null; then
  echo "ERROR: rendered command contains ${_FORBID1} or ${_FORBID2}"
  exit 9
fi

echo "=== Stream L NAM56R 7/7 MIMO + DSA 9+4 + FP8 indexer (${VARIANT}) ==="
echo "RUN_ID=${RUN_ID} VARIANT=${VARIANT} TP=${TP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE} EP=${EP_SIZE} MBS=${MBS} GBS=${GBS} MTP=${MTP_DEPTHS}"
echo "DSA_A_LAYER_RANKS=${CPPMEGA_DSA_A_LAYER_RANKS}"
echo "CPPMEGA_DSA_INDEXER_DTYPE=${CPPMEGA_DSA_INDEXER_DTYPE}"
echo "LOG=${LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

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
  --save "${CKPT_DIR}" \
  --load "${CKPT_DIR}" \
  --save-interval 1000000 \
  --log-interval 1 \
  > "${LOG}" 2>&1

EXIT_CODE=$?
echo "=== Exit code: ${EXIT_CODE} ==="
echo "=== Last 80 lines of ${LOG} ==="
tail -80 "${LOG}"
exit ${EXIT_CODE}
