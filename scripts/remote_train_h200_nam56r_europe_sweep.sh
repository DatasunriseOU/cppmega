#!/usr/bin/env bash
# NAM56R europe H200x8 parameter sweep — PP/EP/topology axes.
#
# 8 sequential configs (E1-E8), 50 iterations each, measure tok/sec at
# iter 30-50. Stop early on OOM, move to next config.
#
# All configs: TP=1, DSA 9+4 (env vars), bf16 indexer, sparse attention
# (tilelang→gather_scatter fallback), real data, --enable-dsa,
# --dsa-indexer-topk 256, --dsa-indexer-loss-coeff 0.001.
#
# Machine: europe h200_1 (LOCATION_2)
# Stack: torch 2.12 nightly, megatron-core 0.18, TE 2.13, cuDNN 92000
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
SWEEP_DIR="${REMOTE_ROOT}/cppmega/nam56r_europe_sweep"
RESULTS_FILE="${SWEEP_DIR}/results_summary.txt"
mkdir -p "${SWEEP_DIR}"

# Activate venv
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"

# cuDNN / NCCL libs
_CL="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/cudnn/lib"
_NL="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/nccl/lib"
_BL="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/cublas/lib"
[ -d "${_CL}" ] && export LD_LIBRARY_PATH="${_CL}:${_NL}:${_BL}:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"

# NAM56R env defaults — constant across all 8 configs
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
export CPPMEGA_DSA_INDEXER_DTYPE="fp8"
export CPPMEGA_DSA_SPARSE_MODE="tilelang"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "${TRITON_CACHE_DIR}"

# Pre-flight
python -c 'import cppmega, megatron; print("cppmega", cppmega.__version__)'

# =====================================================================
# Config definitions: E1-E8
# Format: NAME PP VPP MBS GBS MTP RECOMPUTE_MODE CUDA_GRAPH_MODE EP
# RECOMPUTE_MODE: "moe_act" or "full"
# CUDA_GRAPH_MODE: "per_module" or "off"
# =====================================================================

declare -a CONFIGS=(
  "E1 2 2 4 64 2 moe_act per_module 2"
  "E2 2 2 4 64 2 moe_act per_module 4"
  "E3 1 1 4 64 2 moe_act per_module 1"
  "E4 1 1 4 64 2 moe_act per_module 2"
  "E5 4 1 4 64 2 moe_act per_module 1"
  "E6 2 2 4 64 2 full off 1"
  "E7 2 2 4 64 2 moe_act per_module 1"
  "E8 2 2 4 32 2 moe_act per_module 1"
)

SEQ_LEN=4096
TRAIN_ITERS=50

# Build workdir with pretrain_mamba shim (shared across all configs)
WORKDIR=$(mktemp -d /tmp/cppmega-nam56r-europe-sweep.XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Shim: install MIMO __post_init__ hook + Mamba3 fp32-bias forward pre-hook + DSA FP8 + sparse."""
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

# (5) DSA indexer path (bf16-only after FP8 indexer removal)
_dsa_dtype = os.environ.get("CPPMEGA_DSA_INDEXER_DTYPE", "bf16")
print(f"[cppmega_mimo_shim] CPPMEGA_DSA_INDEXER_DTYPE resolves to '{_dsa_dtype}'")
if _dsa_dtype == "fp8":
    raise RuntimeError(
        "CPPMEGA_DSA_INDEXER_DTYPE=fp8 is no longer supported: dsa_fp8_patch.py "
        "and dsa_fp8_indexer.py were deleted on 2026-04-13. Use bf16."
    )
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

echo "=== Workdir: ${WORKDIR} ==="
echo "=== Starting europe sweep: $(date) ==="
echo "" > "${RESULTS_FILE}"
echo "europe production DSA 9+4 parameter sweep (2026-04-12)" >> "${RESULTS_FILE}"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"

# =====================================================================
# Helper: run a single config
# =====================================================================
run_config() {
    local NAME="$1" PP="$2" VPP="$3" MBS="$4" GBS="$5" MTP="$6" RECOMPUTE="$7" CG_MODE="$8" EP="$9"
    local RUN_ID="nam56r_europe_${NAME}"
    local LOG="${SWEEP_DIR}/${RUN_ID}.log"
    local CKPT_DIR="${SWEEP_DIR}/${RUN_ID}_ckpt"

    mkdir -p "${CKPT_DIR}"
    rm -rf "${CKPT_DIR}"/* || true

    echo ""
    echo "================================================================"
    echo " Config ${NAME}: PP=${PP} VPP=${VPP} MBS=${MBS} GBS=${GBS} MTP=${MTP} RECOMPUTE=${RECOMPUTE} CG=${CG_MODE} EP=${EP}"
    echo " Started: $(date)"
    echo "================================================================"

    # Build hybrid layer pattern
    local HYBRID_PATTERN
    HYBRID_PATTERN=$(python - <<PYEOF
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern

mtp_depths = ${MTP}
pp = ${PP}
vpp = ${VPP}
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
PYEOF
    )
    echo "  HYBRID_PATTERN: ${HYBRID_PATTERN}"

    # Build native args (use build_megatron_args_bundle directly so we can
    # pass dsa_indexer_topk and dsa_indexer_loss_coeff)
    local NATIVE_ARGS
    NATIVE_ARGS=$(python - <<PYEOF
from cppmega.recipes.megatron_args import build_megatron_args_bundle
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan

mtp_depths = ${MTP}
enable_mtp = mtp_depths > 0
plan = build_nam56r_feature_plan(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=max(mtp_depths, 1))
bundle = build_megatron_args_bundle(
    plan=plan,
    use_mla=True,
    use_mtp=enable_mtp,
    mtp_mode="hybrid",
    mtp_num_predictors=mtp_depths,
    use_moe=True,
    moe_expert_model_parallel_size=${EP},
    use_dsa=True,
    dsa_indexer_topk=256,
    dsa_indexer_loss_coeff=0.001,
    dsa_indexer_dtype="bf16",
)
print(bundle.to_shell_fragment())
PYEOF
    )


    echo "  NATIVE_ARGS: ${NATIVE_ARGS}"

    # CUDA graph flags
    local CG_FLAGS=""
    case "${CG_MODE}" in
        off)
            CG_FLAGS="--cuda-graph-impl none"
            ;;
        per_module)
            CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess --cuda-graph-warmup-steps 3"
            ;;
    esac

    # Recompute flags
    local RECOMPUTE_FLAGS=""
    case "${RECOMPUTE}" in
        moe_act)
            RECOMPUTE_FLAGS="--recompute-granularity selective --recompute-modules moe_act"
            ;;
        full)
            RECOMPUTE_FLAGS="--recompute-granularity selective --recompute-modules transformer_layer"
            ;;
    esac

    # MoE dispatcher + capacity
    local MOE_EXTRA_FLAGS="--moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0"

    # No-rope-fusion for PP>1
    local ROPE_FLAG=""
    if [ "${PP}" -gt 1 ]; then
        ROPE_FLAG="--no-rope-fusion"
    fi

    echo "  CG_FLAGS: ${CG_FLAGS}"
    echo "  RECOMPUTE_FLAGS: ${RECOMPUTE_FLAGS}"
    echo "  ROPE_FLAG: ${ROPE_FLAG}"
    echo "  LOG: ${LOG}"

    # Kill any lingering training processes
    pkill -f "pretrain_mamba" 2>/dev/null || true
    sleep 2

    # Confirm GPUs are free
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

    # Run training (timeout 15 min per config — ample for 50 iters)
    local START_TIME=$(date +%s)
    timeout 900 python -m torch.distributed.run --nproc_per_node=8 "${WORKDIR}/pretrain_mamba.py" \
        --data-path 1.0 "${REMOTE_ROOT}/data/megatron/clang_semantic_4k_v10_train" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "${REMOTE_ROOT}/tokenizer" \
        --split 98,1,1 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size ${PP} \
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
        ${CG_FLAGS} \
        ${RECOMPUTE_FLAGS} \
        ${MOE_EXTRA_FLAGS} \
        --no-check-for-nan-in-loss-and-grad \
        ${NATIVE_ARGS} \
        --save "${CKPT_DIR}" \
        --load "${CKPT_DIR}" \
        --save-interval 1000000 \
        --log-interval 1 \
        > "${LOG}" 2>&1
    local EXIT_CODE=$?
    local END_TIME=$(date +%s)
    local ELAPSED=$(( END_TIME - START_TIME ))

    echo "  Exit code: ${EXIT_CODE} (${ELAPSED}s elapsed)"

    # Extract metrics
    local STATUS="OK"
    local TOK_SEC="—"
    local ITER_MS="—"
    local PEAK_GB="—"
    local LOSS_50="—"

    if [ ${EXIT_CODE} -ne 0 ]; then
        if grep -q "OutOfMemoryError" "${LOG}" 2>/dev/null; then
            STATUS="OOM"
        elif [ ${EXIT_CODE} -eq 124 ]; then
            STATUS="TIMEOUT"
        else
            STATUS="CRASH(${EXIT_CODE})"
        fi
        echo "  STATUS: ${STATUS}"
        echo "  Last 20 lines of log:"
        tail -20 "${LOG}" 2>/dev/null || true
    else
        # Extract iter times for iters 30-50 (skip warmup)
        # Megatron log format: " iteration       30/ ... | elapsed time per iteration (ms): XXXX.X | ..."
        ITER_MS=$(grep -oP 'iteration\s+(\d+).*?elapsed time per iteration \(ms\): \K[\d.]+' "${LOG}" | tail -20 | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "—"}')

        if [ "${ITER_MS}" != "—" ] && [ -n "${ITER_MS}" ]; then
            TOK_SEC=$(python3 -c "print(int(${GBS} * ${SEQ_LEN} * 1000 / ${ITER_MS}))")
        fi

        # Extract peak memory (look for "max allocated" or memory reporting lines)
        PEAK_GB=$(grep -oP 'max allocated:\s*[\d.]+' "${LOG}" | tail -1 | grep -oP '[\d.]+' || echo "—")
        if [ "${PEAK_GB}" = "—" ] || [ -z "${PEAK_GB}" ]; then
            # Fallback: look for memory reporting in the log
            PEAK_GB=$(grep -oP 'allocated:\s*[\d.]+ GiB' "${LOG}" | tail -1 | grep -oP '[\d.]+' || echo "—")
        fi

        # Extract loss at final iteration
        LOSS_50=$(grep -oP 'lm loss:\s*\K[\d.E+-]+' "${LOG}" | tail -1 || echo "—")
        if [ -z "${LOSS_50}" ]; then
            LOSS_50="—"
        fi

        STATUS="OK"
    fi

    echo "  RESULTS: tok/sec=${TOK_SEC} iter_ms=${ITER_MS} peak_GB=${PEAK_GB} loss@50=${LOSS_50} status=${STATUS}"

    # Append to results file
    echo "${NAME} | ${PP} | ${VPP} | ${MBS} | ${GBS} | ${MTP} | ${RECOMPUTE} | ${CG_MODE} | ${EP} | ${TOK_SEC} | ${ITER_MS} | ${PEAK_GB} | ${LOSS_50} | ${STATUS}" >> "${RESULTS_FILE}"

    # Clean up checkpoint dir to save space
    rm -rf "${CKPT_DIR}" 2>/dev/null || true

    return ${EXIT_CODE}
}

# =====================================================================
# Main sweep loop
# =====================================================================

echo "| Config | PP | VPP | MBS | GBS | MTP | Recompute | CG | EP | tok/sec | iter_ms | peak_GB | loss@50 | status |" >> "${RESULTS_FILE}"
echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|" >> "${RESULTS_FILE}"

for config_line in "${CONFIGS[@]}"; do
    read -r NAME PP VPP MBS GBS MTP RECOMPUTE CG_MODE EP <<< "${config_line}"
    run_config "${NAME}" "${PP}" "${VPP}" "${MBS}" "${GBS}" "${MTP}" "${RECOMPUTE}" "${CG_MODE}" "${EP}" || true
    echo ""
done

echo ""
echo "================================================================"
echo " Sweep complete: $(date)"
echo "================================================================"
echo ""
echo "=== RESULTS TABLE ==="
cat "${RESULTS_FILE}"
echo ""
echo "All logs in: ${SWEEP_DIR}/"
