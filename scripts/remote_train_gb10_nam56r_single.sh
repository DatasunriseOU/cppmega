#!/usr/bin/env bash
# Launch NAM56R training on GB10 (NVIDIA Grace Blackwell consumer, sm_121)
# using a single-GPU configuration.  This is the first NAM56R run on consumer
# Blackwell -- the H200 noconv launcher is TP=1/PP=4 (needs 4 GPUs minimum),
# whereas GB10 is a single unified Grace+Blackwell package with 1 GPU and
# 128 GB unified memory.
#
# Key differences from remote_train_h200_nam56r_noconv.sh:
#   - Direct ssh (NOT gcloud compute ssh) -- GB10 is not a GCP instance
#   - nproc_per_node=1, PP=1, no sequence-parallel
#   - seq-length 2048, MBS=1, GBS=1
#   - GB10 attention backend is profile-controlled.  Equal-dim attention can
#     force FA2, but NAM56R MLA uses TE auto/fused because the current FA4 CUTE
#     wheel rejects sm_121 and FA2 does not support MLA.
#   - No checkpoint save (avoid sharded_state_dict): --save-interval 50000000
#   - cppmega ngram_hash and structure features disabled for the smoke
#
# Pre-requisite: Phase 0 merge flag patch on installed mamba_ssm package
# (``TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`` in the mamba3_mimo_*
# pass_configs).  Already applied by the driver before this script was run.
#
# Usage:
#
#     CPPMEGA_TRAIN_ITERS=5 \
#     bash scripts/remote_train_gb10_nam56r_single.sh
#
# CUDA graph tuning (2026-04-10):
#   - CPPMEGA_GB10_USE_DIST_OPT=0 (default) drops --use-distributed-optimizer;
#     on a single GPU there is nothing to shard and the extra optimizer stream
#     triggers the "AccumulateGrad node's stream does not match" warning every
#     iter and prevents CUDA graph capture.
#   - CPPMEGA_GB10_CUDA_GRAPH=1 (default) adds:
#        --cuda-graph-impl local --cuda-graph-warmup-steps 3
#     (matches the known-good fragment emitted by nam56r_nemo_recipe.py on
#     H200, which does NOT use a separate --enable-cuda-graph gate).
#     `local` is the only impl that works with the noconv Mamba3 stack; the
#     transformer_engine impl hits a MoE shared_experts.cached_fc2_input assert
#     (see memory/project_nam56r_throughput.md).  `--moe-shared-expert-overlap`
#     is intentionally NOT added -- it is incompatible with local graphs.
#   - CPPMEGA_GB10_SILENCE_ACC_GRAD=1 (default) silences PyTorch's
#     AccumulateGrad stream-mismatch warning via a small runtime shim so the
#     log stays scannable.  Set to 0 to see the warning again.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-gb10}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dave}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_nam56r_gb10_single}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-nam56r-gb10-single.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-gb10-nam56r.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"
export CPPMEGA_ATTN_BACKEND="${CPPMEGA_ATTN_BACKEND:-auto}"
case "${CPPMEGA_ATTN_BACKEND}" in
  auto|flash|fused|unfused) ;;
  *)
    echo "FATAL: CPPMEGA_ATTN_BACKEND must be auto, flash, fused, or unfused" >&2
    exit 2
    ;;
esac
mkdir -p "${REMOTE_ROOT}/cppmega" "${REMOTE_ROOT}/.triton-cache"
REMOTE_CKPT_DIR="${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}_ckpt"
REMOTE_WORKDIR="$(mktemp -d /tmp/cppmega-nam56r-gb10.XXXXXX)"
trap 'rm -rf "${REMOTE_WORKDIR}"' EXIT

python -c "import cppmega, megatron, transformer_engine; print('import smoke ok', cppmega.__version__)"
python -c "from cppmega.megatron.nam56r_noconv_spec import build_cppmega_nam56r_noconv_stack_spec; print('noconv spec importable')"

# GB10 sm_121 smem-cap preflight: every in-tree TileLang kernel must declare
# TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE.  On sm_121 this is a HARD fail.
# See cppmega/megatron/preflight_smem_check.py and reference_gb10_bwd_bwd_blocker.md.
python -m cppmega.megatron.preflight_smem_check

cp "${REMOTE_ROOT}/megatron-lm/pretrain_mamba.py" "${REMOTE_WORKDIR}/pretrain_mamba_inner.py"

# Wrapper that applies a small pre-flight patch for the GB10 single-GPU CUDA
# graph path and then hands control to the upstream pretrain_mamba.py.  The
# wrapper lives in the same workdir so the sys.argv passthrough is transparent.
cat > "${REMOTE_WORKDIR}/pretrain_mamba.py" <<'PY'
"""GB10 single-GPU CUDA-graph pre-flight wrapper around pretrain_mamba."""
from __future__ import annotations

import os
import runpy
import sys
import warnings

import torch

# (1) Silence the AccumulateGrad stream-mismatch warning.  The warning is
# emitted every iteration on single-GPU when the optimizer and forward are
# enqueued on different streams; it is *not* actionable from the launch script
# (the fix lives inside Megatron's DDP init) and its presence adds per-iter
# Python overhead plus a log-line flood that hides CUDA graph capture messages.
if os.environ.get("CPPMEGA_GB10_SILENCE_ACC_GRAD", "1") == "1":
    setter = getattr(
        torch.autograd.graph,
        "set_warn_on_accumulate_grad_stream_mismatch",
        None,
    )
    if setter is not None:
        try:
            setter(False)
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"[gb10-wrapper] failed to disable acc-grad warn: {exc}", flush=True)
    # Belt & braces: also filter by substring in case the setter is unavailable
    # on this PyTorch build.
    warnings.filterwarnings(
        "ignore",
        message=r".*AccumulateGrad node's stream does not match.*",
    )

# (2) Hand off to the real pretrain_mamba.py.  We use runpy so the upstream
# file runs under `__main__` exactly as `python pretrain_mamba.py` would.
_here = os.path.dirname(os.path.abspath(__file__))
_inner = os.path.join(_here, "pretrain_mamba_inner.py")
sys.argv[0] = _inner
runpy.run_path(_inner, run_name="__main__")
PY

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
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CPPMEGA_NEM_PATTERN="${CPPMEGA_NEM_PATTERN:-AEMEAEMEAEMR}"
export CPPMEGA_LAYER_DEPTH="${CPPMEGA_LAYER_DEPTH:-52}"
export CPPMEGA_R_LAYER_INDICES="${CPPMEGA_R_LAYER_INDICES:-12,24,36,48}"
export CPPMEGA_NGRAM_HASH_ENABLED="${CPPMEGA_NGRAM_HASH_ENABLED:-0}"
export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CPPMEGA_OPTIMIZER="muon"
export CPPMEGA_MUON_NUM_NS_STEPS="${CPPMEGA_MUON_NUM_NS_STEPS:-3}"
export CPPMEGA_MUON_SCALAR_OPTIMIZER="adam8bit"
export CPPMEGA_MUON_QUANTIZED_MOMENTUM="${CPPMEGA_MUON_QUANTIZED_MOMENTUM:-1}"
export CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE="${CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE:-int8}"
export CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER=1
export CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER=1
export CPPMEGA_GRAD_REDUCE_IN_BF16=1

# Start nvidia-smi logger in background for peak-mem capture
NVSMI_LOG="${REMOTE_LOG%.log}.nvsmi.log"
( while true; do
    ts="$(date '+%Y-%m-%dT%H:%M:%S')"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits \
      | while IFS=, read -r mu mt ug tg; do echo "${ts},${mu},${mt},${ug},${tg}"; done
    sleep 2
  done ) > "${NVSMI_LOG}" 2>&1 &
NVSMI_PID=$!
cleanup() { rm -rf "${REMOTE_WORKDIR}"; kill "${NVSMI_PID}" 2>/dev/null || true; }
trap cleanup EXIT

HYBRID_LAYER_PATTERN="$(${REMOTE_VENV}/bin/python - <<PY
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern
print(build_default_hybrid_layer_pattern(mtp_depths=1))
PY
)"

NATIVE_ARGS="$(${REMOTE_VENV}/bin/python - <<PY
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan

plan = build_nam56r_feature_plan(pattern='${CPPMEGA_NEM_PATTERN}', depth=${CPPMEGA_LAYER_DEPTH}, mtp_depths=1)
bundle = build_nam56r_megatron_native_args(
    plan=plan,
    enable_mla=True,
    enable_mtp=True,
    mtp_mode='hybrid',
    enable_moe=${CPPMEGA_ENABLE_MOE:-True},
)
print(bundle.to_shell_fragment())
PY
)"

_CPPMEGA_DATA_PATH_TOKENS="${CPPMEGA_DATA_PATH:-1.0 /home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10_train}"
# shellcheck disable=SC2206
DATA_ARGS=(
  --data-path ${_CPPMEGA_DATA_PATH_TOKENS}
  --tokenizer-type "${CPPMEGA_TOKENIZER_TYPE:-HuggingFaceTokenizer}"
  --tokenizer-model "${CPPMEGA_TOKENIZER_MODEL:-/home/dave/cppmega-root/cpp_tokenizer_hf}"
  --vocab-size "${CPPMEGA_VOCAB_SIZE:-65536}"
  --make-vocab-size-divisible-by 128
)

# Single-GPU knobs: --use-distributed-optimizer is pointless on nproc=1 and
# causes an extra optimizer stream that breaks CUDA graph capture.  Kept
# behind an env var so the old behavior is reproducible.
DIST_OPT_ARGS=()
if [ "${CPPMEGA_GB10_USE_DIST_OPT:-0}" = "1" ]; then
  DIST_OPT_ARGS+=(--use-distributed-optimizer)
fi

OPTIMIZER_ARGS=(
  --optimizer "${CPPMEGA_OPTIMIZER}"
  --muon-momentum 0.95
  --muon-scale-mode spectral
  --muon-num-ns-steps "${CPPMEGA_MUON_NUM_NS_STEPS}"
  --muon-tp-mode blockwise
  --muon-scalar-optimizer "${CPPMEGA_MUON_SCALAR_OPTIMIZER}"
  --use-bf16-no-master-emerging-optimizer
  --use-bf16-no-master-emerging-fallback-optimizer
  --grad-reduce-in-bf16
)
if [ "${CPPMEGA_MUON_QUANTIZED_MOMENTUM}" = "1" ]; then
  OPTIMIZER_ARGS+=(
    --muon-quantized-momentum
    --muon-quantized-momentum-dtype "${CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE}"
  )
fi

# CUDA graph capture.  NAM56R uses DROPLESS MoE (no capacity factor, no
# pad-expert-input-to-capacity) so the MoE layer has dynamic shapes and
# cannot be fully captured in a CUDA graph -- attempting to do so hits
# "RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
# capture" inside MoEAlltoAll dispatcher line 295.
#
# Use --cuda-graph-impl=transformer_engine --cuda-graph-scope=attn which
# captures ONLY TransformerLayer._forward_attention() and leaves the MoE
# _forward_mlp() uncaptured.  This matches the bench3 H200 noconv launcher.
#
# Modes:
#   te_attn (default) - TE impl, attention-only scope, dropless MoE friendly
#   none              - CUDA graphs disabled (fallback)
CPPMEGA_GB10_CUDA_GRAPH="${CPPMEGA_GB10_CUDA_GRAPH:-te_attn}"
CUDA_GRAPH_ARGS=()
case "${CPPMEGA_GB10_CUDA_GRAPH}" in
  te_attn|1)
    CUDA_GRAPH_ARGS+=(
      --cuda-graph-impl transformer_engine
      --cuda-graph-scope attn
      --cuda-graph-warmup-steps "${CPPMEGA_GB10_CUDA_GRAPH_WARMUP:-3}"
    )
    ;;
  te_wide)
    # Wide scope: capture attention + mamba scan + MoE router preprocess,
    # but NOT the dropless MoE dispatch itself (dynamic shapes) and NOT mlp
    # (NAM56R has no dense MLP layers — all feed-forward is MoE, so scoping
    # `mlp` hits an AssertionError "mlp cuda graph is only supported for
    # dense layers, but not found in the model").  Expected 3-5x speedup
    # vs te_attn-only by capturing the mamba chunk scan which runs eagerly
    # and adds 10-15 seconds per iteration through Python launch overhead.
    CUDA_GRAPH_ARGS+=(
      --cuda-graph-impl transformer_engine
      --cuda-graph-scope attn mamba moe_router moe_preprocess
      --cuda-graph-warmup-steps "${CPPMEGA_GB10_CUDA_GRAPH_WARMUP:-3}"
    )
    ;;
  none|0)
    CUDA_GRAPH_ARGS+=(--cuda-graph-impl none)
    ;;
  local_full)
    # Kept for reference -- will crash on dropless MoE.
    CUDA_GRAPH_ARGS+=(
      --cuda-graph-impl local
      --cuda-graph-scope full_iteration
      --cuda-graph-warmup-steps "${CPPMEGA_GB10_CUDA_GRAPH_WARMUP:-3}"
      --no-check-for-nan-in-loss-and-grad
    )
    ;;
  *)
    echo "Unknown CPPMEGA_GB10_CUDA_GRAPH=${CPPMEGA_GB10_CUDA_GRAPH}; expected te_attn|te_wide|none|local_full" >&2
    exit 2
    ;;
esac

# Optional nsys profiling wrapper.  Gated on CPPMEGA_GB10_NSYS_PROFILE=1.  We
# skip the first 30 s of training (import + iter-1 compile) then capture 30 s
# of steady state.  Keep --sample=none so the CPU profile does not dominate;
# we only care about GPU kernel accounting.
NSYS_PREFIX=()
if [ "${CPPMEGA_GB10_NSYS_PROFILE:-0}" = "1" ]; then
  NSYS_OUTPUT="${CPPMEGA_GB10_NSYS_OUTPUT:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}_nsys}"
  NSYS_DELAY="${CPPMEGA_GB10_NSYS_DELAY:-30}"
  NSYS_DURATION="${CPPMEGA_GB10_NSYS_DURATION:-30}"
  NSYS_PREFIX=(
    nsys profile
    --trace=cuda,nvtx
    --sample=none
    --cpuctxsw=none
    --cuda-graph-trace=node
    --cuda-flush-interval=50
    --delay="${NSYS_DELAY}"
    --force-overwrite=true
    --output="${NSYS_OUTPUT}"
  )
  if [ "${CPPMEGA_GB10_NSYS_DURATION:-0}" != "0" ]; then
    NSYS_PREFIX+=(--duration="${NSYS_DURATION}")
  fi
  echo "[gb10-nsys] wrapping python with: ${NSYS_PREFIX[*]}"
fi

# MoE token dispatcher override.  Megatron default is 'allgather', which on a
# single GPU broadcasts every token to every expert and then filters — this is
# the slowest option.  `alltoall` does direct per-expert placement and on a
# single GPU still runs but avoids the redundant all-gather.  Exposed as
# CPPMEGA_GB10_MOE_DISPATCHER so we can compare against baseline.
EXTRA_MOE_ARGS=()
if [ -n "${CPPMEGA_GB10_MOE_DISPATCHER:-}" ]; then
  EXTRA_MOE_ARGS+=(--moe-token-dispatcher-type "${CPPMEGA_GB10_MOE_DISPATCHER}")
fi

# Optional FP8 (hybrid) toggle.  CPPMEGA_GB10_FP8=1 flips on the standard TE
# FP8 hybrid format.  Ported from the H200 fp8 matrix script.  Mamba3 fp32
# bias/D/dt tensors are preserved via the cppmega_fp8_shim one-shot Float16Module
# patch that ships in scripts/cppmega_fp8_shim.py.  On GB10 (sm_121a) TE FP8
# uses the Blackwell-consumer path: no tcgen05, no TMEM, but the cuBLASLt FP8
# GEMM on cuBLAS 13.2 is officially tuned for Spark per release notes.
FP8_ARGS=()
if [ "${CPPMEGA_GB10_FP8:-0}" = "1" ]; then
  FP8_ARGS+=(
    --fp8-format hybrid
    --fp8-amax-history-len 16
    --fp8-amax-compute-algo max
  )
fi

"${NSYS_PREFIX[@]}" python -m torch.distributed.run --nproc_per_node=1 "${REMOTE_WORKDIR}/pretrain_mamba.py" \
  "${DATA_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  "${DIST_OPT_ARGS[@]}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${EXTRA_MOE_ARGS[@]}" \
  "${FP8_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}" \
  --hidden-size "${CPPMEGA_HIDDEN_SIZE:-3584}" \
  --ffn-hidden-size "${CPPMEGA_FFN_HIDDEN_SIZE:-18944}" \
  --num-attention-heads "${CPPMEGA_NUM_ATTN_HEADS:-28}" \
  --seq-length "${CPPMEGA_SEQ_LENGTH:-2048}" \
  --max-position-embeddings "${CPPMEGA_MAX_POSITION_EMBEDDINGS:-2048}" \
  --micro-batch-size "${CPPMEGA_MICRO_BATCH_SIZE:-1}" \
  --global-batch-size "${CPPMEGA_GLOBAL_BATCH_SIZE:-1}" \
  --train-iters "${CPPMEGA_TRAIN_ITERS:-5}" \
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
  --attention-backend "${CPPMEGA_ATTN_BACKEND}" \
  --spec cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec \
  ${NATIVE_ARGS} \
  --save "${REMOTE_CKPT_DIR}" \
  --save-interval 50000000 \
  --log-interval 1 \
  > "${REMOTE_LOG}" 2>&1

tail -n 120 "${REMOTE_LOG}"
echo "--- nvsmi peak ---"
awk -F, '{ if ($2+0 > peak) peak=$2+0 } END { print "peak_used_mib="peak }' "${NVSMI_LOG}"
INNER

scp "${LOCAL_TMP_SCRIPT}" "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"
ssh "${REMOTE_HOST}" "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' CPPMEGA_TRAIN_ITERS='${CPPMEGA_TRAIN_ITERS:-5}' CPPMEGA_SEQ_LENGTH='${CPPMEGA_SEQ_LENGTH:-2048}' CPPMEGA_MAX_POSITION_EMBEDDINGS='${CPPMEGA_MAX_POSITION_EMBEDDINGS:-2048}' CPPMEGA_MICRO_BATCH_SIZE='${CPPMEGA_MICRO_BATCH_SIZE:-1}' CPPMEGA_GLOBAL_BATCH_SIZE='${CPPMEGA_GLOBAL_BATCH_SIZE:-1}' CPPMEGA_NGRAM_HASH_ENABLED='${CPPMEGA_NGRAM_HASH_ENABLED:-0}' CPPMEGA_STRUCTURE_ENABLED='${CPPMEGA_STRUCTURE_ENABLED:-0}' CPPMEGA_GB10_USE_DIST_OPT='${CPPMEGA_GB10_USE_DIST_OPT:-0}' CPPMEGA_GB10_CUDA_GRAPH='${CPPMEGA_GB10_CUDA_GRAPH:-1}' CPPMEGA_GB10_CUDA_GRAPH_WARMUP='${CPPMEGA_GB10_CUDA_GRAPH_WARMUP:-3}' CPPMEGA_GB10_SILENCE_ACC_GRAD='${CPPMEGA_GB10_SILENCE_ACC_GRAD:-1}' CPPMEGA_MUON_NUM_NS_STEPS='${CPPMEGA_MUON_NUM_NS_STEPS:-3}' CPPMEGA_NEM_PATTERN='${CPPMEGA_NEM_PATTERN:-AEMEAEMEAEMR}' CPPMEGA_LAYER_DEPTH='${CPPMEGA_LAYER_DEPTH:-52}' CPPMEGA_R_LAYER_INDICES='${CPPMEGA_R_LAYER_INDICES:-12,24,36,48}' CPPMEGA_HIDDEN_SIZE='${CPPMEGA_HIDDEN_SIZE:-3584}' CPPMEGA_FFN_HIDDEN_SIZE='${CPPMEGA_FFN_HIDDEN_SIZE:-18944}' CPPMEGA_NUM_ATTN_HEADS='${CPPMEGA_NUM_ATTN_HEADS:-28}' CPPMEGA_ENABLE_MOE='${CPPMEGA_ENABLE_MOE:-True}' CPPMEGA_ATTN_BACKEND='${CPPMEGA_ATTN_BACKEND:-auto}' CPPMEGA_GB10_NSYS_PROFILE='${CPPMEGA_GB10_NSYS_PROFILE:-0}' CPPMEGA_GB10_NSYS_OUTPUT='${CPPMEGA_GB10_NSYS_OUTPUT:-}' CPPMEGA_GB10_NSYS_DELAY='${CPPMEGA_GB10_NSYS_DELAY:-30}' CPPMEGA_GB10_NSYS_DURATION='${CPPMEGA_GB10_NSYS_DURATION:-30}' CPPMEGA_GB10_MOE_DISPATCHER='${CPPMEGA_GB10_MOE_DISPATCHER:-}' CPPMEGA_GB10_FP8='${CPPMEGA_GB10_FP8:-0}' CPPMEGA_DATA_PATH='${CPPMEGA_DATA_PATH:-}' CPPMEGA_TOKENIZER_TYPE='${CPPMEGA_TOKENIZER_TYPE:-HuggingFaceTokenizer}' CPPMEGA_TOKENIZER_MODEL='${CPPMEGA_TOKENIZER_MODEL:-/home/dave/cppmega-root/cpp_tokenizer_hf}' CPPMEGA_VOCAB_SIZE='${CPPMEGA_VOCAB_SIZE:-65536}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'remote nam56r gb10 single-gpu train failed; tail follows:'; tail -n 200 '${REMOTE_LOG}' || true; exit \$status)"
ssh "${REMOTE_HOST}" "rm -f '${REMOTE_TMP_SCRIPT}'"
