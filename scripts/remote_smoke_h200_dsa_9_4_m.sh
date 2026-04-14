#!/usr/bin/env bash
# Stream M: NAM56R DSA 9+4 + FP8 indexer + loss_coeff==0 gate + EP sweep
#           + selective MoE recompute.
#
# Inherits Stream L topology (TP=1 PP=2 VPP=2 MBS=4 GBS=64 MTP=2) but adds:
#   1. loss_coeff==0 gate in dsa_fp8_patch.py (commit 26ff3ca) saves ~63 GB
#   2. --recompute-granularity selective --recompute-modules moe_act
#
# VARIANT=v1 -> EP=4 DP=1 (4 experts/rank, gradient accumulation handles GBS)
# VARIANT=v2 -> EP=2 DP=2 (8 experts/rank, 2x data-parallel)
#
# NO feature disable hacks. NO MBS drop. NO DSA off. NO MTP off. NO MoE off.
# NO CppmegaMamba3TPMixer. Real data only. Mamba-3 MIMO 7/7 preserved.
#
# Designed to run directly on LOCATION_2 (h200_1)
# inside a tmux session launched with `bash -l`. No gcloud/scp wrapping here
# -- this script IS the remote body.
set -euo pipefail

# Source .bashrc for cuDNN/NCCL/cublas LD_LIBRARY_PATH -- bash -l in tmux
# does NOT source .bashrc on this machine.
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"

REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
VARIANT="${VARIANT:-v1}"
RUN_ID="${RUN_ID:-cppmega_nam56r_dsa_9_4_fp8_m_${VARIANT}}"
LOG="${LOG:-${REMOTE_ROOT}/cppmega/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-${REMOTE_ROOT}/cppmega/nam56r_grid/${RUN_ID}_ckpt}"

mkdir -p "$(dirname "${LOG}")" "${CKPT_DIR}"
rm -rf "${CKPT_DIR}"/* || true  # fresh init each run

# Activate venv
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"

# venv NVIDIA libs must come first (cuDNN for MLA, NCCL for comms, cublas for MoE).
# bash -l on europe does NOT source .bashrc, so we must set this explicitly.
_NV="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia"
_LD_PREFIX=""
for _pkg in cudnn nccl cublas; do
  _d="${_NV}/${_pkg}/lib"
  [ -d "${_d}" ] && _LD_PREFIX="${_d}:${_LD_PREFIX}"
done
export LD_LIBRARY_PATH="${_LD_PREFIX}/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"

# NAM56R env defaults (same as Stream D v2 / Stream J / Stream L)
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
# DSA indexer dtype: lemyx handles this, no separate FP8 patch needed
# TileLang SparseMLA (default): fused online-softmax sparse attention kernel
# from Megatron-LM PR #3674. The kernel is parameterized for arbitrary d_v
# (verified working with d_v=512 on H200). ~40% throughput improvement over
# gather_scatter. Monkey-patched via cppmega_fp8_shim.py patch (7).
export CPPMEGA_DSA_SPARSE_MODE="${CPPMEGA_DSA_SPARSE_MODE:-tilelang}"

# Mamba/M2RNN activation checkpointing — saves ~28+ GiB per microbatch
# by recomputing SSM forward during backward instead of saving intermediates.
export CPPMEGA_MAMBA_RECOMPUTE="${CPPMEGA_MAMBA_RECOMPUTE:-1}"
# Skip indexer loss (7 GiB torch.bmm) until head-streaming loss is implemented.
export CPPMEGA_DSA_SKIP_INDEXER_LOSS="${CPPMEGA_DSA_SKIP_INDEXER_LOSS:-1}"
export CPPMEGA_DSA_INDEXER_LOSS_COEFF="${CPPMEGA_DSA_INDEXER_LOSS_COEFF:-0}"
export CPPMEGA_SELECTIVE_FP8_MOE="${CPPMEGA_SELECTIVE_FP8_MOE:-0}"
export CPPMEGA_MTP_LIGER_CE="${CPPMEGA_MTP_LIGER_CE:-0}"
# CPPMEGA_MAMBA3_COMPILE removed — regional compile is now always-on
export CPPMEGA_LEMYX_DSA="${CPPMEGA_LEMYX_DSA:-0}"
# FP8 sparse attention: when 1, unfused_dsa_fn dispatches to SparseMLA_FP8
# which does per-token quantize bf16->fp8 then runs the FP8 TileLang kernel.
# Also overrides dsa.SparseMLA so _fused_sparse_mla_absorbed uses FP8.
export CPPMEGA_DSA_FP8_ATTENTION="${CPPMEGA_DSA_FP8_ATTENTION:-0}"
# IndexCache: cross-layer index reuse for DSA (skip indexer on 6/9 layers)
export CPPMEGA_INDEX_CACHE="${CPPMEGA_INDEX_CACHE:-0}"
# Always on — prevents fragmentation OOM on large models
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
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
# Variant overrides (v1 = EP=4 DP=1, v2 = EP=2 DP=2)
# ---------------------------------------------------------------------------
case "${VARIANT}" in
  v0)
    echo "[stream_m] V0 = EP=1 DP=4 PP=2 VPP=2 (all 16 experts/rank, 4x DP, no DeepEP)"
    EP_SIZE=1
    # EP=1: flex dispatcher requires TP*EP > 1, so use alltoall for routing.
    # But do NOT disable CG — alltoall at EP=1 is just local routing, CG-safe.
    CPPMEGA_DISPATCHER_OVERRIDE=alltoall_keep_cg
    ;;
  v1)
    echo "[stream_m] V1 = EP=4 DP=1 PP=2 VPP=2 (4 experts/rank, grad-accum GBS)"
    EP_SIZE=4
    ;;
  v2)
    echo "[stream_m] V2 = EP=2 DP=2 PP=2 VPP=2 (8 experts/rank, 2x DP)"
    EP_SIZE=2
    ;;
  *)
    echo "ERROR: unknown VARIANT=${VARIANT} (expected v0|v1|v2)" >&2
    exit 2
    ;;
esac

# Build workdir with pretrain_mamba shim (MIMO __post_init__ + fp32-bias hook
# + DSA FP8 patch + loss_coeff==0 gate + peak-memory reporter).
WORKDIR=$(mktemp -d /tmp/cppmega-nam56r-dsa-9-4-m.XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Shim: install MIMO __post_init__ hook + Mamba3 fp32-bias forward pre-hook
+ Stream E/G DSA FP8 fwd+bwd patch + loss_coeff==0 gate + per-rank
peak-memory reporter + cuDNN fused-attn workaround for europe."""
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

# (3) Mamba3 fp32-bias forward pre-hook
try:
    from mamba_ssm.modules.mamba3 import Mamba3 as _Mamba3
    import torch as _torch
    # Mamba3 fp32 bias: Float16Module casts D/dt_bias/B_bias/C_bias to bf16,
    # but TileLang MIMO kernel expects DT (from dt_bias) as float32.
    # The upstream forward computes DT = F.softplus(dd_dt + self.dt_bias) which
    # returns fp32 only if dt_bias is fp32. We must restore fp32 on params.
    # CG-safe: use is_graph_capturing() guard to skip during graph capture.
    # Mamba3 fp32 fix: TileLang MIMO kernel expects DT, ADT, D, biases as fp32.
    # Float16Module._apply casts ALL params to bf16 AFTER __init__.
    # Fix: override Float16Module._apply to skip Mamba3 fp32 params.
    # This is a ONE-TIME fix at model construction, not per-forward.
    # Guard Mamba3._apply to preserve fp32 params when Float16Module casts to bf16.
    # ONLY dt_bias and D need fp32 — they feed TileLang fp32 params DT/D.
    # B_bias/C_bias are ADDED to bf16 Q/K tensors → must stay bf16.
    # mimo_x/z/o are used in .float() einsum → auto-promoted, no issue.
    # _apply guard REMOVED: keeping D/dt_bias as fp32 while rest is bf16 causes
    # distributed optimizer save crash — optimizer state tensors get cross-mapped
    # between the bf16 and fp32 gradient buffers (model_param_group_index_map
    # ordering mismatch). Instead, let Float16Module cast D/dt_bias to bf16 normally
    # and use .float() in forward (already done in mamba_ssm Mamba3 + noconv_mamba_mixer).
    print("[cppmega_mimo_shim] Mamba3._apply guard DISABLED (bf16 D/dt_bias + .float() in fwd)")
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

# (5) Stream E+G: DSA FP8 fwd+bwd indexer patch + loss_coeff==0 gate
# DSA FP8 patch removed — lemyx + IndexCache is the only path

# (6) Mamba/M2RNN activation checkpointing (CPPMEGA_MAMBA_RECOMPUTE=1)
_mamba_recompute = os.environ.get("CPPMEGA_MAMBA_RECOMPUTE", "0") == "1"
if _mamba_recompute:
    try:
        from cppmega.megatron.mamba_recompute_patch import apply_mamba_recompute_patch
        _applied = apply_mamba_recompute_patch()
        print(f"[cppmega_mimo_shim] Mamba recompute patch applied={_applied}")
    except Exception as _exc:
        print(f"[cppmega_mimo_shim] Mamba recompute patch failed: {_exc}", file=sys.stderr)
        raise

# (8) Selective FP8 for MoE only (CPPMEGA_SELECTIVE_FP8_MOE=1)
if os.environ.get("CPPMEGA_SELECTIVE_FP8_MOE", "0") == "1":
    try:
        from cppmega.megatron.selective_fp8_moe_patch import apply_selective_fp8_moe_patch
        apply_selective_fp8_moe_patch()
    except Exception as _exc:
        print(f"[cppmega_mimo_shim] Selective FP8 MoE patch failed: {_exc}", file=sys.stderr)
        raise

# (11) Override dsa_indexer_loss_coeff via env var
_ilc = os.environ.get("CPPMEGA_DSA_INDEXER_LOSS_COEFF")
if _ilc is not None:
    from megatron.core.transformer.transformer_config import TransformerConfig
    _old_post = TransformerConfig.__post_init__
    _ilc_val = float(_ilc)
    def _ilc_post_init(self):
        _old_post(self)
        if hasattr(self, "dsa_indexer_loss_coeff"):
            object.__setattr__(self, "dsa_indexer_loss_coeff", _ilc_val)
        if hasattr(self, "dsa_indexer_use_sparse_loss"):
            _sparse = os.environ.get("CPPMEGA_DSA_SPARSE_LOSS", "1") == "1"
            object.__setattr__(self, "dsa_indexer_use_sparse_loss", _sparse)
    TransformerConfig.__post_init__ = _ilc_post_init
    print(f"[cppmega_mimo_shim] dsa_indexer_loss_coeff overridden to {_ilc_val}")

# (13) lemyx fused DSA warmup kernel (CPPMEGA_LEMYX_DSA=1)
if os.environ.get("CPPMEGA_LEMYX_DSA", "0") == "1":
    from cppmega.megatron.lemyx_dsa_warmup import apply_lemyx_dsa_patch
    apply_lemyx_dsa_patch()

# (12) IndexCache: cross-layer index reuse (CPPMEGA_INDEX_CACHE=1)
if os.environ.get("CPPMEGA_INDEX_CACHE", "0") == "1":
    from cppmega.megatron.index_cache_patch import apply_index_cache_patch
    apply_index_cache_patch()

# (14) Hybrid schedule plan for combined_1f1b EP A2A overlap
if os.environ.get("CPPMEGA_HYBRID_SCHEDULE_PLAN", "0") == "1":
    try:
        from cppmega.megatron.hybrid_schedule_plan import apply_hybrid_schedule_plan_patch
        apply_hybrid_schedule_plan_patch()
    except Exception as _exc:
        print(f"[cppmega_mimo_shim] Hybrid schedule plan patch failed: {_exc}", file=sys.stderr)
        raise

# (15) TileLang SparseMLA monkey-patch for DSA sparse attention
_sparse_mode = os.environ.get("CPPMEGA_DSA_SPARSE_MODE", "tilelang").strip().lower()
_fp8_attn = os.environ.get("CPPMEGA_DSA_FP8_ATTENTION", "0") == "1"
if _sparse_mode not in ("gather_scatter", "gather-scatter", "pytorch"):
    try:
        from megatron.core.transformer.experimental_attention_variant import dsa as _dsa_mod
        _existing_unfused = getattr(_dsa_mod, "unfused_dsa_fn", None)
        if _existing_unfused is not None and not getattr(
            _existing_unfused, "__cppmega_sparse_dsa_patched__", False
        ):
            if _fp8_attn:
                from cppmega.megatron.sparse_mla_ops.sparse_mla import (
                    sparse_mla_fp8_as_unfused_dsa as _sparse_mla_fn,
                )
                _label = "FP8"
            else:
                from cppmega.megatron.sparse_mla_ops.sparse_mla import (
                    sparse_mla_as_unfused_dsa as _sparse_mla_fn,
                )
                _label = "BF16"
            setattr(_sparse_mla_fn, "__cppmega_sparse_dsa_patched__", True)
            _dsa_mod.unfused_dsa_fn = _sparse_mla_fn
            # If FP8 attention, also override SparseMLA in dsa module namespace
            # so _fused_sparse_mla_absorbed always uses FP8 (even for BF16 inputs).
            if _fp8_attn:
                from cppmega.megatron.sparse_mla_ops.sparse_mla import SparseMLA_FP8 as _FP8Cls
                _dsa_mod.SparseMLA = _FP8Cls
                print("[cppmega_mimo_shim] dsa.SparseMLA overridden with SparseMLA_FP8 (always-on FP8)")
            print(
                f"[cppmega_mimo_shim] TileLang SparseMLA {_label} applied "
                "(replaces unfused_dsa_fn: fused online-softmax sparse attention)"
            )
    except Exception as _exc:
        print(f"[cppmega_mimo_shim] TileLang SparseMLA patch failed: {_exc}", file=sys.stderr)
        # Fall back to gather_scatter
        try:
            from megatron.core.transformer.experimental_attention_variant import dsa as _dsa_mod
            from cppmega.megatron.dsa_sparse_attention import sparse_dsa_fn as _sparse_dsa_fn
            setattr(_sparse_dsa_fn, "__cppmega_sparse_dsa_patched__", True)
            _dsa_mod.unfused_dsa_fn = _sparse_dsa_fn
            print("[cppmega_mimo_shim] Fallback: gather_scatter sparse_dsa_fn applied")
        except Exception as _exc2:
            print(f"[cppmega_mimo_shim] gather_scatter fallback also failed: {_exc2}", file=sys.stderr)
else:
    try:
        from megatron.core.transformer.experimental_attention_variant import dsa as _dsa_mod
        from cppmega.megatron.dsa_sparse_attention import sparse_dsa_fn as _sparse_dsa_fn
        _existing_unfused = getattr(_dsa_mod, "unfused_dsa_fn", None)
        if _existing_unfused is not None and not getattr(
            _existing_unfused, "__cppmega_sparse_dsa_patched__", False
        ):
            setattr(_sparse_dsa_fn, "__cppmega_sparse_dsa_patched__", True)
            _dsa_mod.unfused_dsa_fn = _sparse_dsa_fn
            print(
                "[cppmega_mimo_shim] gather_scatter sparse_dsa_fn applied "
                "(CPPMEGA_DSA_SPARSE_MODE=gather_scatter)"
            )
    except Exception as _exc:
        print(f"[cppmega_mimo_shim] gather_scatter patch failed: {_exc}", file=sys.stderr)

# (10) Mamba3 regional torch.compile — always on, no env gate
from cppmega.megatron.mamba3_compile_patch import apply_mamba3_compile_patch
apply_mamba3_compile_patch()

# (9) MTP Liger fused CE (CPPMEGA_MTP_LIGER_CE=1)
if os.environ.get("CPPMEGA_MTP_LIGER_CE", "0") == "1":
    from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
    patch_mtp_loss_with_liger()

# (7) Stream M: per-rank peak-memory reporter (atexit hook)
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
            f"[stream_m_peak_mem] rank={rank} device={dev} "
            f"peak_alloc_gib={peak_alloc:.3f} peak_reserved_gib={peak_reserved:.3f}",
            flush=True,
        )
    except Exception as _exc:
        print(f"[stream_m_peak_mem] report failed: {_exc}", file=sys.stderr)

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
    assert total % n_chunks == 0, f"cannot split {total}-layer main into {n_chunks} equal chunks"
    chunks = [main[i*per:(i+1)*per] for i in range(n_chunks)]
    main = "|".join(chunks)
print(main + (("/" + mtp_part) if mtp_part else ""))
PY
)
echo "HYBRID_PATTERN: ${HYBRID_PATTERN}"

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
    enable_moe=True,
    enable_dsa=True,
)
print(bundle.to_shell_fragment())
PY
)
echo "NATIVE_ARGS (pre-sed): ${NATIVE_ARGS}"

# Override --mtp-num-layers to match MTP_DEPTHS if >1.
if [ "${MTP_DEPTHS}" -gt 1 ]; then
  NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/--mtp-num-layers 1/--mtp-num-layers ${MTP_DEPTHS}/")
fi

# Strip --dsa-indexer-dtype from NATIVE_ARGS: it is a cppmega recipe arg, not
# a Megatron CLI arg. The FP8 indexer behavior is controlled entirely by the
# CPPMEGA_DSA_INDEXER_DTYPE env var and the shim's apply_dsa_fp8_patch().
NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed 's/ *--dsa-indexer-dtype [a-z0-9]*//')

# Override --expert-model-parallel-size to EP_SIZE (always runs for Stream M:
# both variants set EP_SIZE >=2).
if [ "${EP_SIZE}" != "1" ]; then
  NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/--expert-model-parallel-size 1/--expert-model-parallel-size ${EP_SIZE}/")
fi

# Confirm the substitution actually happened.
if ! echo "${NATIVE_ARGS}" | grep -q -- "--expert-model-parallel-size ${EP_SIZE}"; then
  echo "ERROR: failed to set --expert-model-parallel-size ${EP_SIZE}; NATIVE_ARGS=${NATIVE_ARGS}" >&2
  exit 3
fi
# EP=1 override: flex dispatcher requires TP*EP > 1, fall back to alltoall
if [[ "${CPPMEGA_DISPATCHER_OVERRIDE:-}" == alltoall* ]]; then
  NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/--moe-token-dispatcher-type flex/--moe-token-dispatcher-type alltoall/")
  NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/ --moe-router-dtype fp32//")
  MOE_EXTRA_FLAGS="--moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0"
  echo "NATIVE_ARGS (alltoall override for EP=1): dispatcher=alltoall"
fi
# Override dsa-indexer-loss-coeff in native args if env var set
if [ "${CPPMEGA_DSA_INDEXER_LOSS_COEFF}" != "" ] && [ "${CPPMEGA_DSA_INDEXER_LOSS_COEFF}" != "0.001" ]; then
  NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/--dsa-indexer-loss-coeff [0-9.e-]*/--dsa-indexer-loss-coeff ${CPPMEGA_DSA_INDEXER_LOSS_COEFF}/")
  echo "NATIVE_ARGS: dsa-indexer-loss-coeff overridden to ${CPPMEGA_DSA_INDEXER_LOSS_COEFF}"
fi
echo "NATIVE_ARGS (post-sed): ${NATIVE_ARGS}"

# CUDA graphs — full scope verified stable with lr warmup.
if [ -z "${CG_FLAGS+x}" ]; then
  CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess"
elif [ "${CG_FLAGS}" = "NONE" ]; then
  CG_FLAGS=""
fi
echo "[stream_m] CUDA graphs: ${CG_FLAGS}"
# NOTE: --moe-pad-expert-input-to-capacity is INCOMPATIBLE with flex dispatcher
# (raises ValueError). Omit when using DeepEP flex.
# Only reset MOE_EXTRA_FLAGS if not already set by alltoall override above.
MOE_EXTRA_FLAGS="${MOE_EXTRA_FLAGS:-}"

# FP8 GEMM flags: enables FP8 for ALL TE linear layers (MLP, MoE experts,
# attention projections, embeddings). --bf16 remains the base precision;
# FP8 applies inside TE's Linear forward via delayed scaling.
if [ -z "${FP8_FLAGS+x}" ]; then
  # FP8_FLAGS not set at all → use default FP8 with tensorwise recipe
  FP8_FLAGS="--fp8-format hybrid --fp8-recipe tensorwise --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
elif [ "${FP8_FLAGS}" = "NONE" ]; then
  FP8_FLAGS=""
fi

# CPPMEGA_FP8_PARAM_GATHER=1 adds --fp8-param-gather: stores param gather bucket
# in FP8 storage (master weights stay FP32). Saves ~5 GiB on NAM56R-class models
# (removes BF16 param-gather copy in dist-optimizer). Tensorwise + dist-opt
# compatible; master stays FP32; custom bf16 modules (TileLang SparseMLA, Mamba3)
# untouched since they are created outside fp8_model_init() context.
# Requires --use-distributed-optimizer (already set below). INCOMPATIBLE with
# --use-precision-aware-optimizer and --optimizer-cpu-offload.
if [ "${CPPMEGA_FP8_PARAM_GATHER:-0}" = "1" ] && [ -n "${FP8_FLAGS}" ]; then
  FP8_FLAGS="${FP8_FLAGS} --fp8-param-gather"
  echo "[stream_m] FP8 param-gather enabled (-5 GiB all-gather buffer)"
fi
# If FP8_FLAGS="" (explicitly empty) → no FP8 (BF16 only)

# Stream M: selective activation recompute.
# head-streaming in dsa_fp8_indexer.py (commit 563fcb0) reduces DSA target from
# 7.5 GiB to 0.8 GiB per layer, so selective moe_act recompute is sufficient.
# --mla-down-proj-fusion: fuses MLA down-projection GEMMs (PR #3039, free perf).
# NOTE: moe_act recompute is INCOMPATIBLE with FP8 *delayed* scaling only.
# FP8 tensorwise (current scaling) IS compatible with moe_act recompute.
# Only drop moe_act if FP8 + delayed recipe.
if echo "${FP8_FLAGS}" | grep -q "fp8" && ! echo "${FP8_FLAGS}" | grep -q "tensorwise"; then
  RECOMPUTE_MODULES="mlp mla_up_proj"
  echo "[stream_m] FP8 delayed → dropping moe_act from recompute"
else
  RECOMPUTE_MODULES="moe_act mlp mla_up_proj"
  echo "[stream_m] moe_act recompute enabled (BF16 or FP8 tensorwise)"
fi
# EP A2A overlap: --overlap-moe-expert-parallel-comm + --delay-wgrad-compute
# Requires CPPMEGA_HYBRID_SCHEDULE_PLAN=1 (set automatically when enabled).
# CUDA_DEVICE_MAX_CONNECTIONS must be >1 for multi-stream overlap.
EP_OVERLAP_FLAGS=""
if [ "${CPPMEGA_EP_OVERLAP:-0}" = "1" ]; then
  EP_OVERLAP_FLAGS="--overlap-moe-expert-parallel-comm --delay-wgrad-compute"
  export CPPMEGA_HYBRID_SCHEDULE_PLAN=1
  export CUDA_DEVICE_MAX_CONNECTIONS=32
  echo "[stream_m] EP A2A overlap enabled (CUDA_DEVICE_MAX_CONNECTIONS=32)"
fi

EXTRA_FLAGS="${EXTRA_FLAGS:---recompute-granularity selective --recompute-modules ${RECOMPUTE_MODULES} --mla-down-proj-fusion --clip-grad 1.0 ${EP_OVERLAP_FLAGS}}"

ROPE_FLAG=""
if [ "${NO_ROPE_FUSION}" = "1" ]; then
  ROPE_FLAG="--no-rope-fusion"
fi

python -c 'import cppmega, megatron; print("cppmega", cppmega.__version__)'
# dsa_fp8_patch/indexer removed — lemyx + IndexCache is the only path

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
if echo "${NATIVE_ARGS} ${CG_FLAGS} ${MOE_EXTRA_FLAGS} ${ROPE_FLAG} ${EXTRA_FLAGS}" | grep -E "(${_FORBID1}|${_FORBID2})" > /dev/null; then
  echo "ERROR: rendered command contains ${_FORBID1} or ${_FORBID2}"
  exit 9
fi

echo "=== Stream M NAM56R 7/7 MIMO + DSA 9+4 + FP8 indexer + loss gate + selective recompute (${VARIANT}) ==="
echo "RUN_ID=${RUN_ID} VARIANT=${VARIANT} TP=${TP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE} EP=${EP_SIZE} MBS=${MBS} GBS=${GBS} MTP=${MTP_DEPTHS}"
echo "DSA_A_LAYER_RANKS=${CPPMEGA_DSA_A_LAYER_RANKS}"
echo "DSA path: lemyx + IndexCache"
echo "CPPMEGA_DSA_FP8_ATTENTION=${CPPMEGA_DSA_FP8_ATTENTION}"
echo "CPPMEGA_INDEX_CACHE=${CPPMEGA_INDEX_CACHE}"
echo "CPPMEGA_MTP_LIGER_CE=${CPPMEGA_MTP_LIGER_CE}"
echo "EXTRA_FLAGS=${EXTRA_FLAGS}"
echo "LOG=${LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

# nsys profiling wrapper
NSYS_CMD=""
if [ "${NSYS_PROFILE:-0}" = "1" ]; then
  NSYS_OUT="${NSYS_OUT:-/home/dave/cppmega-root/cppmega/nsys_${RUN_ID}}"
  NSYS_CMD="nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true -o ${NSYS_OUT} -f true --stats=true --duration=${NSYS_DURATION:-300}"
  echo "[nsys] profiling 120s → ${NSYS_OUT}.nsys-rep"
fi

# Memory debug: dumps detailed CUDA allocation info after step 0
if [ "${CPPMEGA_MEMORY_DEBUG:-0}" = "1" ]; then
  export NANOCHAT_MEMORY_DEBUG=1
  echo "[mem_debug] CUDA memory snapshot enabled (step 0)"
fi

${NSYS_CMD} python -m torch.distributed.run --nproc_per_node=8 "${WORKDIR}/pretrain_mamba.py" \
  --data-path 1.0 "${REMOTE_ROOT}/data/megatron/clang_semantic_4k_v10_train" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${REMOTE_ROOT}/data/tokenizer" \
  --split 98,1,1 \
  --tensor-model-parallel-size ${TP_SIZE} \
  --pipeline-model-parallel-size ${PP_SIZE} \
  --context-parallel-size 1 \
  --sequence-parallel \
  ${OPTIMIZER_FLAGS:---use-distributed-optimizer} \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  ${ROPE_FLAG} \
  --hybrid-layer-pattern "${HYBRID_PATTERN}" \
  --hidden-size 4096 \
  --ffn-hidden-size 21504 \
  --num-attention-heads 32 \
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
  ${FP8_FLAGS} \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --attention-backend auto \
  --spec cppmega.megatron.nam56r_full_spec build_cppmega_nam56r_full_stack_spec \
  ${CG_FLAGS} \
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
exit ${EXIT_CODE}
