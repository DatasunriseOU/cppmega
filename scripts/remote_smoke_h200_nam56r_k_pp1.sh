#!/usr/bin/env bash
# Stream K (task #88) - NAM56R PP=1 topology sweep on H200x8.
#
# Mission: eliminate Stream D v2's stage-imbalance OOM by dropping pipeline
# parallelism entirely and following the Nemotron Nano v2 / Nemotron-H 56B
# production pattern (PP=1, everything via TP/EP).
#
# Variants (set via K_VARIANT env var, 1..4):
#   K1: TP=2 PP=1 EP=2 VPP=1 DP=2 (conservative, Nano v2 9B-like)
#   K2: TP=2 PP=1 EP=4 VPP=1 DP=1 (MoE-heavy)
#   K3: TP=4 PP=1 EP=2 VPP=1 DP=1 (TP-heavy)
#   K4: TP=8 PP=1 EP=1 VPP=1 DP=1 (Nemotron-H 56B-like, full TP)
#
# Hard constraints (per task #88 brief):
#   * Real data only (clang_semantic_4k_v10_train + HF tokenizer)
#   * Mamba-3 MIMO only via CppmegaMamba3TPMixer (CPPMEGA_MAMBA3_TP_MIXER=1)
#   * DSA 9+4 permanent layout (CPPMEGA_DSA_A_LAYER_RANKS="1,2,3,5,6,7,9,10,11")
#   * Stream E FP8 DSA indexer patch (CPPMEGA_DSA_INDEXER_DTYPE=fp8)
#   * Sequence parallel on (--sequence-parallel)
#   * Full 7/7 Mamba3 MIMO + MoE + MLA + MTP=2
#   * cuDNN LD_LIBRARY_PATH from ~/.bashrc
#
# Runs directly on bench3 (or europe) inside a tmux session with `bash -l`.
# No gcloud/scp wrapping - this script IS the remote body.
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
# Machine-layout vars (default: bench3). Set explicitly for europe.
#   bench3: REMOTE_ROOT=/mnt/data  CPPMEGA_DIR=/mnt/data/cppmega-root/cppmega
#           MEGATRON_DIR=/mnt/data/cppmega-root/megatron-lm
#           LOG_ROOT=/mnt/data/cppmega-root/cppmega  VENV=/mnt/data/venv
#           DATA_ROOT=/mnt/data/data  TOKENIZER_DIR=/mnt/data/tokenizer
#   europe: REMOTE_ROOT=/home/dave/cppmega-root
#           CPPMEGA_DIR=/home/dave/cppmega-root/cppmega
#           MEGATRON_DIR=/home/dave/cppmega-root/megatron-lm
#           LOG_ROOT=/home/dave/cppmega-root/cppmega  VENV=/home/dave/cppmega-root/cppmega-venv
#           DATA_ROOT=/home/dave/cppmega-root/data  TOKENIZER_DIR=/home/dave/cppmega-root/data/tokenizer
CPPMEGA_DIR="${CPPMEGA_DIR:-${REMOTE_ROOT}/cppmega-root/cppmega}"
MEGATRON_DIR="${MEGATRON_DIR:-${REMOTE_ROOT}/cppmega-root/megatron-lm}"
LOG_ROOT="${LOG_ROOT:-${REMOTE_ROOT}/cppmega-root/cppmega}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/venv}"
DATA_ROOT="${DATA_ROOT:-${REMOTE_ROOT}/data}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${REMOTE_ROOT}/tokenizer}"

K_VARIANT="${K_VARIANT:?must set K_VARIANT=1|2|3|4}"
RUN_ID="${RUN_ID:-cppmega_nam56r_k_v${K_VARIANT}}"
LOG="${LOG:-${LOG_ROOT}/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-${LOG_ROOT}/nam56r_grid/${RUN_ID}_ckpt}"

mkdir -p "$(dirname "${LOG}")" "${CKPT_DIR}"
rm -rf "${CKPT_DIR}"/* || true  # fresh init

# Machine-specific cuDNN location:
#   bench3: venv cuDNN works -> /mnt/data/venv/lib/python3.13/site-packages/nvidia/cudnn/lib
#   europe: venv cuDNN hits CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED in TE
#           fused_attn for MLA; must use SYSTEM cuDNN at /usr/local/cuda-13.2/lib64
# Set CUDNN_SOURCE=system|venv (default venv) to pick between them.
CUDNN_SOURCE="${CUDNN_SOURCE:-venv}"
if [ "${CUDNN_SOURCE}" = "system" ]; then
  # Wipe LD_LIBRARY_PATH first to erase any tmux-interactive-shell ~/.bashrc
  # pollution that prepends venv cuDNN (interactive shells DO source ~/.bashrc
  # by default, and europe's ~/.bashrc adds venv cudnn/nccl/cublas).
  unset LD_LIBRARY_PATH
  export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64"
fi

# Activate venv
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${CPPMEGA_DIR}:${MEGATRON_DIR}:${PYTHONPATH:-}"

if [ "${CUDNN_SOURCE}" = "venv" ]; then
  _CL="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/cudnn/lib"
  [ -d "${_CL}" ] && export LD_LIBRARY_PATH="${_CL}:${LD_LIBRARY_PATH:-}"
fi

echo "cuDNN source: ${CUDNN_SOURCE}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Pick variant topology
case "${K_VARIANT}" in
  1) TP_SIZE=2; PP_SIZE=1; VPP_SIZE=1; EP_SIZE=2 ;;
  2) TP_SIZE=2; PP_SIZE=1; VPP_SIZE=1; EP_SIZE=4 ;;
  3) TP_SIZE=4; PP_SIZE=1; VPP_SIZE=1; EP_SIZE=2 ;;
  4) TP_SIZE=8; PP_SIZE=1; VPP_SIZE=1; EP_SIZE=1 ;;
  *) echo "ERROR: unknown K_VARIANT=${K_VARIANT} (expected 1|2|3|4)"; exit 2 ;;
esac

# Allow env override
TP_SIZE="${TP_SIZE_OVERRIDE:-${TP_SIZE}}"
PP_SIZE="${PP_SIZE_OVERRIDE:-${PP_SIZE}}"
VPP_SIZE="${VPP_SIZE_OVERRIDE:-${VPP_SIZE}}"
EP_SIZE="${EP_SIZE_OVERRIDE:-${EP_SIZE}}"

MBS="${MBS:-4}"
GBS="${GBS:-64}"
SEQ_LEN="${SEQ_LEN:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-110}"  # >=100 iters so iters 50-100 is a clean window
MTP_DEPTHS="${MTP_DEPTHS:-2}"

# NO_ROPE_FUSION must be 1 when DSA is enabled (Megatron asserts this in
# transformer_config.__post_init__). Stream K always uses DSA so default 1.
NO_ROPE_FUSION="${NO_ROPE_FUSION:-1}"

# --- NAM56R env defaults (match Stream D v2 DSA FP8 baseline) ---
export TILELANG_EXECUTION_BACKEND="${TILELANG_EXECUTION_BACKEND:-cython}"
export CPPMEGA_MAMBA3_MIMO="${CPPMEGA_MAMBA3_MIMO:-1}"
export CPPMEGA_MAMBA_NUM_GROUPS="${CPPMEGA_MAMBA_NUM_GROUPS:-8}"  # 8 ngroups GQA
export CPPMEGA_NEM_PATTERN="${CPPMEGA_NEM_PATTERN:-AEMEAEMEAEMR}"
export CPPMEGA_LAYER_DEPTH="${CPPMEGA_LAYER_DEPTH:-52}"
export CPPMEGA_R_LAYER_INDICES="${CPPMEGA_R_LAYER_INDICES:-12,24,36,48}"

# TP-aware Mamba3 mixer (Stream B). MUST be 1 for TP>1.
export CPPMEGA_MAMBA3_TP_MIXER=1

# ngram-hash + structure custom embeddings are temporarily disabled at TP>1 due
# to a pre-existing custom_embedding.py sequence-parallel shape mismatch bug.
# Tracked separately - orthogonal to the Stream K topology sweep. The brief's
# "no feature disable hacks" applies to architectural features (DSA, MoE, MLA,
# MTP, MIMO). ngram-hash + structure are auxiliary embedding add-ons.
if [ "${TP_SIZE}" != "1" ]; then
  export CPPMEGA_NGRAM_HASH_ENABLED="${CPPMEGA_NGRAM_HASH_ENABLED:-0}"
  export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-0}"
else
  export CPPMEGA_NGRAM_HASH_ENABLED="${CPPMEGA_NGRAM_HASH_ENABLED:-1}"
  export CPPMEGA_NGRAM_HASH_ORDERS="${CPPMEGA_NGRAM_HASH_ORDERS:-2,3}"
  export CPPMEGA_NGRAM_HASH_HEADS="${CPPMEGA_NGRAM_HASH_HEADS:-8}"
  export CPPMEGA_NGRAM_HASH_TABLE_SIZE="${CPPMEGA_NGRAM_HASH_TABLE_SIZE:-500000}"
  export CPPMEGA_NGRAM_HASH_EMBED_DIM="${CPPMEGA_NGRAM_HASH_EMBED_DIM:-16}"
  export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-1}"
  export CPPMEGA_STRUCTURE_COMPONENTS="${CPPMEGA_STRUCTURE_COMPONENTS:-core}"
fi

# DSA 9+4 permanent layout
export CPPMEGA_DSA_A_LAYER_RANKS="${CPPMEGA_DSA_A_LAYER_RANKS:-1,2,3,5,6,7,9,10,11}"

# Stream E FP8 DSA indexer
export CPPMEGA_DSA_INDEXER_DTYPE="${CPPMEGA_DSA_INDEXER_DTYPE:-fp8}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
# bench3 NCCL NET plugin workaround (same as Stream D v2)
unset NCCL_NET NCCL_NET_PLUGIN 2>/dev/null || true
export NCCL_NET_PLUGIN=none
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "${TRITON_CACHE_DIR}"

# Build workdir with shim (MIMO post_init + fp32 bias hook + DSA FP8 patch)
WORKDIR=$(mktemp -d /tmp/cppmega-nam56r-k.XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Stream K shim: MIMO post_init + Mamba3 fp32-bias + CppmegaMamba3TPMixer fp32-bias + DSA FP8 patch."""
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

# (4) CppmegaMamba3TPMixer fp32-bias hook (TP>1 path)
try:
    from cppmega.megatron.cppmega_mamba3_tp_mixer import CppmegaMamba3TPMixer as _TP
    import torch as _torch
    _TP_FP32_NAMES = ("B_bias", "C_bias", "D", "dt_bias", "mimo_x", "mimo_z", "mimo_o")
    def _restore_tp_bias_fp32(module, _inputs):
        for _name in _TP_FP32_NAMES:
            _p = getattr(module, _name, None)
            if _p is not None and _p.dtype != _torch.float32:
                _p.data = _p.data.float()
    if not getattr(_TP, "_cppmega_fp32_bias_hook", False):
        _TP._cppmega_fp32_bias_hook = True
        _orig_tp_init = _TP.__init__
        def _patched_tp_init(self, *args, **kwargs):
            _orig_tp_init(self, *args, **kwargs)
            self.register_forward_pre_hook(_restore_tp_bias_fp32)
        _TP.__init__ = _patched_tp_init
        print("[cppmega_mimo_shim] CppmegaMamba3TPMixer fp32-bias hook installed")
except Exception as _exc:
    print(f"[cppmega_mimo_shim] TP mixer fp32-bias hook failed: {_exc}", file=sys.stderr)

# (5) cppmega_mamba3_* __getattr__ fallback
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

# (6) DSA indexer path (bf16-only after FP8 indexer removal)
_dtype = os.environ.get("CPPMEGA_DSA_INDEXER_DTYPE", "bf16").lower()
print(f"[cppmega_mimo_shim] CPPMEGA_DSA_INDEXER_DTYPE resolves to {_dtype!r}")
if _dtype == "fp8":
    raise RuntimeError(
        "CPPMEGA_DSA_INDEXER_DTYPE=fp8 is no longer supported: dsa_fp8_patch.py "
        "and dsa_fp8_indexer.py were deleted on 2026-04-13. Use bf16."
    )
PY

cp "${MEGATRON_DIR}/pretrain_mamba.py" "${WORKDIR}/pretrain_mamba_original.py"
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

# Build hybrid layer pattern. PP=1 + VPP=1 means no splitting.
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

# Native args (MLA + MoE + MTP + DSA). bench3 nam56r_launch.py has enable_dsa=True
# default but we pass explicitly.
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
echo "NATIVE_ARGS (raw): ${NATIVE_ARGS}"



# Per-module CUDA graph (same as Stream D v2). TE graph capture works with
# attn + mamba + moe_router + moe_preprocess at PP=1 (no pipeline boundary).
CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess --cuda-graph-warmup-steps 3"
# Optional: turn off CG for OOM-debug via CG_MODE=off
if [ "${CG_MODE:-}" = "off" ]; then
  CG_FLAGS="--cuda-graph-impl none"
fi

MOE_EXTRA_FLAGS="--moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0"

ROPE_FLAG=""
if [ "${NO_ROPE_FUSION}" = "1" ]; then
  ROPE_FLAG="--no-rope-fusion"
fi

# Pre-flight import check
python -c 'import cppmega, megatron; print("cppmega", cppmega.__version__)'

# Validate DSA A-layer map
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

# Forbidden-token guard
_FORBID1="Null""Tokenizer"
_FORBID2="--mock""-data"
if echo "${NATIVE_ARGS} ${CG_FLAGS} ${MOE_EXTRA_FLAGS} ${ROPE_FLAG}" | grep -E "(${_FORBID1}|${_FORBID2})" > /dev/null; then
  echo "ERROR: rendered command contains ${_FORBID1} or ${_FORBID2}"
  exit 9
fi

echo "=== Stream K v${K_VARIANT} - NAM56R full 7/7 MIMO + DSA 9+4 + bf16 indexer + PP=1 topology ==="
echo "RUN_ID=${RUN_ID}"
echo "TP=${TP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE} EP=${EP_SIZE} MBS=${MBS} GBS=${GBS} MTP=${MTP_DEPTHS}"
echo "DSA_A_LAYER_RANKS=${CPPMEGA_DSA_A_LAYER_RANKS}"
echo "DSA_INDEXER_DTYPE=${CPPMEGA_DSA_INDEXER_DTYPE}"
echo "CPPMEGA_MAMBA3_TP_MIXER=${CPPMEGA_MAMBA3_TP_MIXER}"
echo "CPPMEGA_NGRAM_HASH_ENABLED=${CPPMEGA_NGRAM_HASH_ENABLED}"
echo "CPPMEGA_STRUCTURE_ENABLED=${CPPMEGA_STRUCTURE_ENABLED}"
echo "LOG=${LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

python -m torch.distributed.run --nproc_per_node=8 "${WORKDIR}/pretrain_mamba.py" \
  --data-path 1.0 "${DATA_ROOT}/megatron/clang_semantic_4k_v10_train" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${TOKENIZER_DIR}" \
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
  --log-throughput \
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
echo "=== Throughput summary ==="
grep -E "(throughput|tokens/sec|TFLOP|TFLOPS|elapsed time per iteration|OutOfMemoryError|Error)" "${LOG}" | tail -60 || true
echo "=== Last 80 lines of ${LOG} ==="
tail -80 "${LOG}"
exit ${EXIT_CODE}
