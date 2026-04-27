#!/usr/bin/env bash
# NAM56R full 7/7 MIMO training on LOCATION_2 H200x8 with TP=2.
#
# Mirrors scripts/remote_train_h200_nam56r_grid.sh but flips the parallelism
# to TP=2 PP=1 and toggles CPPMEGA_MAMBA3_TP_MIXER=1 so the
# CppMegaSelectiveMambaMixer wraps CppmegaMamba3TPMixer (TP-aware Mamba3)
# instead of AuthorMamba3Mixer (TP=1 only).
#
# Baseline for comparison: TP=1 PP=2 VPP=2 MBS=4 GBS=64 MTP=2 = 112k tok/sec
# (see docs/nam56r_grid_search_2026_04_12.md, "112k tok/sec baseline" row).
#
# Hard constraints:
#   * Real data only (clang_semantic_4k_v10_train) -- never NullTokenizer
#   * Full 7/7 Mamba3 MIMO features (nam56r_full_spec)
#   * tensor_model_parallel_size=2
#   * cuDNN path fixed via ~/.bashrc on europe; we re-export here as a safety net
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_1}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_2}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
RUN_ID="${RUN_ID:-nam56r_tp2_run}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-nam56r-tp2.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-nam56r-tp2.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
# Pull the cuDNN/NCCL/cuBLAS LD_LIBRARY_PATH lines out of ~/.bashrc directly.
# Sourcing ~/.bashrc would not work because bash's non-interactive guard
# (`case $- in *i*) ;; *) return;; esac`) returns immediately.
: "${LD_LIBRARY_PATH:=}"
# Use the system CUDA 13.2 libraries (cuDNN, NCCL, cuBLAS).  The venv-bundled
# cuDNN under nvidia/cudnn/lib hits CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
# inside TE fused_attn for MLA on this image; the system cuDNN 9.20 in
# /usr/local/cuda-13.2/lib64 is the working configuration.
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH}"
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"
mkdir -p "${REMOTE_ROOT}/cppmega" "${REMOTE_ROOT}/.triton-cache"

CKPT_DIR="${REMOTE_ROOT}/cppmega/${RUN_ID}_ckpt"
WORKDIR="$(mktemp -d /tmp/cppmega-nam56r-tp2.XXXXXX)"
trap 'rm -rf "${WORKDIR}"' EXIT

mkdir -p "$(dirname "${REMOTE_LOG}")" "${CKPT_DIR}"
rm -rf "${CKPT_DIR}"/* || true

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}" >&2

# ---- NAM56R env defaults (mirror grid script) ----
export TILELANG_EXECUTION_BACKEND="${TILELANG_EXECUTION_BACKEND:-cython}"
export CPPMEGA_MAMBA3_MIMO="${CPPMEGA_MAMBA3_MIMO:-1}"
export CPPMEGA_MAMBA_NUM_GROUPS="${CPPMEGA_MAMBA_NUM_GROUPS:-8}"
export CPPMEGA_NEM_PATTERN="${CPPMEGA_NEM_PATTERN:-AEMEAEMEAEMR}"
export CPPMEGA_LAYER_DEPTH="${CPPMEGA_LAYER_DEPTH:-52}"
export CPPMEGA_R_LAYER_INDICES="${CPPMEGA_R_LAYER_INDICES:-12,24,36,48}"
# NOTE: ngram-hash + structure custom embeddings are temporarily disabled at
# TP=2 because cppmega/megatron/custom_embedding.py does not handle
# sequence-parallel sharding (the ngram embeddings come out at full L while
# the parallel embedding is sliced to L/tp -- shape mismatch in the add).
# This is a pre-existing bug in custom_embedding.py orthogonal to the
# CppmegaMamba3TPMixer task; tracked separately.
export CPPMEGA_NGRAM_HASH_ENABLED="${CPPMEGA_NGRAM_HASH_ENABLED:-0}"
export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "${TRITON_CACHE_DIR}"

# *** TP-aware Mamba3 mixer toggle ***
export CPPMEGA_MAMBA3_TP_MIXER=1

# ---- Parallelism knobs (TP=2 default; everything else mirrors baseline) ----
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
VPP_SIZE="${VPP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
MBS="${MBS:-4}"
GBS="${GBS:-64}"
SEQ_LEN="${SEQ_LEN:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
MTP_DEPTHS="${MTP_DEPTHS:-2}"
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-per_module}"
FP8_MODE="${FP8_MODE:-off}"
NO_ROPE_FUSION="${NO_ROPE_FUSION:-0}"  # only needed when PP>1

# ---- Build the same shim as the grid launcher ----
cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Shim: install MIMO __post_init__ hook + Mamba3 fp32-bias forward pre-hook."""
from __future__ import annotations
import os
import sys

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

# Also install the same fp32 forward pre-hook on CppmegaMamba3TPMixer so its
# per-head Mamba3 params stay fp32 even when wrapped in Float16Module.
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

# ---- Build hybrid layer pattern (no PP/VPP splitting since PP=1, VPP=1) ----
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
if mtp_depths == 0:
    mtp_part = ""
n_chunks = pp * max(vpp, 1)
if n_chunks > 1:
    total = len(main)
    per = total // n_chunks
    assert total % n_chunks == 0
    chunks = [main[i*per:(i+1)*per] for i in range(n_chunks)]
    main = "|".join(chunks)
print(main + (("/" + mtp_part) if mtp_part else ""))
PY
)
echo "HYBRID_PATTERN: ${HYBRID_PATTERN}"

# ---- Native arg fragment (MLA + MoE + MTP) ----
NATIVE_ARGS=$(python - <<PY
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan
mtp_depths = ${MTP_DEPTHS}
enable_mtp = mtp_depths > 0
plan = build_nam56r_feature_plan(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=max(mtp_depths, 1))
bundle = build_nam56r_megatron_native_args(
    plan=plan, enable_mla=True, enable_mtp=enable_mtp,
    mtp_mode="hybrid", mtp_num_predictors=mtp_depths, enable_moe=True,
    moe_expert_model_parallel_size=${EP_SIZE},
)
print(bundle.to_shell_fragment())
PY
)
echo "NATIVE_ARGS: ${NATIVE_ARGS}"


# CUDA graph flags
CG_FLAGS=""
case "${CUDA_GRAPH_MODE}" in
  off)
    CG_FLAGS="--cuda-graph-impl none"
    ;;
  per_module)
    CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe --cuda-graph-warmup-steps 3"
    ;;
  full_iteration)
    CG_FLAGS="--cuda-graph-impl local --cuda-graph-scope full_iteration --cuda-graph-warmup-steps 3"
    ;;
  *)
    echo "ERROR: unknown CUDA_GRAPH_MODE=${CUDA_GRAPH_MODE}"; exit 2;;
esac

FP8_FLAGS=""
if [ "${FP8_MODE}" = "hybrid_mla_moe" ]; then
  FP8_FLAGS="--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
fi

ROPE_FLAG=""
if [ "${NO_ROPE_FUSION}" = "1" ]; then
  ROPE_FLAG="--no-rope-fusion"
fi

MOE_EXTRA_FLAGS="--moe-token-dispatcher-type alltoall"
if [ "${CUDA_GRAPH_MODE}" != "off" ]; then
  MOE_EXTRA_FLAGS="${MOE_EXTRA_FLAGS} --moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0 --moe-permute-fusion"
fi

python -c 'import cppmega, megatron; print("cppmega", cppmega.__version__)'

if grep -E "^\s*--(tokenizer-type NullTokenizer|mock-data)\b" "$0" > /dev/null 2>&1; then
  echo "ERROR: forbidden NullTokenizer/mock-data tokens detected"
  exit 9
fi

echo "=== NAM56R full 7/7 MIMO TP=${TP_SIZE} run ==="
echo "RUN_ID=${RUN_ID} TP=${TP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE} EP=${EP_SIZE} MBS=${MBS} GBS=${GBS} MTP=${MTP_DEPTHS} CG=${CUDA_GRAPH_MODE} FP8=${FP8_MODE}"
echo "CPPMEGA_MAMBA3_TP_MIXER=${CPPMEGA_MAMBA3_TP_MIXER}"
echo "LOG=${REMOTE_LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true

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
  --log-throughput \
  ${CG_FLAGS} \
  ${FP8_FLAGS} \
  ${MOE_EXTRA_FLAGS} \
  --no-check-for-nan-in-loss-and-grad \
  ${NATIVE_ARGS} \
  --save "${CKPT_DIR}" \
  --load "${CKPT_DIR}" \
  --save-interval 1000000 \
  --log-interval 1 \
  > "${REMOTE_LOG}" 2>&1

EXIT_CODE=$?
echo "=== Exit code: ${EXIT_CODE} ==="
echo "=== Throughput summary ==="
grep -E "(throughput|tokens/sec|TFLOP|TFLOPS|elapsed time per iteration)" "${REMOTE_LOG}" | tail -40 || true
echo "=== Last 60 lines of ${REMOTE_LOG} ==="
tail -60 "${REMOTE_LOG}"
exit ${EXIT_CODE}
INNER

gcloud compute scp --zone "${REMOTE_ZONE}" "${LOCAL_TMP_SCRIPT}" "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"
gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' RUN_ID='${RUN_ID}' TP_SIZE='${TP_SIZE:-2}' PP_SIZE='${PP_SIZE:-1}' VPP_SIZE='${VPP_SIZE:-1}' EP_SIZE='${EP_SIZE:-1}' MBS='${MBS:-4}' GBS='${GBS:-64}' SEQ_LEN='${SEQ_LEN:-4096}' TRAIN_ITERS='${TRAIN_ITERS:-100}' MTP_DEPTHS='${MTP_DEPTHS:-2}' CUDA_GRAPH_MODE='${CUDA_GRAPH_MODE:-per_module}' FP8_MODE='${FP8_MODE:-off}' NO_ROPE_FUSION='${NO_ROPE_FUSION:-0}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'training failed; tail follows:'; tail -n 200 '${REMOTE_LOG}' 2>/dev/null || true; exit \$status)"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "rm -f '${REMOTE_TMP_SCRIPT}'"
