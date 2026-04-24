#!/usr/bin/env bash
# Memory profiler: NAM56R DSA 9+4 on europe H200x8
#
# Stripped-down version of remote_smoke_h200_dsa_9_4_m.sh with:
#   - VARIANT=v0 (EP=1, no DeepEP -- isolate model memory)
#   - TRAIN_ITERS=5 (just enough to see memory pattern)
#   - NO CUDA graphs (see pure memory, no graph overhead)
#   - Comprehensive memory profiling hooks in the shim
#
# Goal: understand the 135 GiB consumption breakdown:
#   model params, optimizer state, activations, framework overhead
set -euo pipefail

[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"

REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
VARIANT="v0"  # EP=1, no DeepEP
RUN_ID="cppmega_nam56r_mem_profile"
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
export CPPMEGA_DSA_INDEXER_DTYPE="${CPPMEGA_DSA_INDEXER_DTYPE:-fp8}"
export CPPMEGA_DSA_SPARSE_MODE="${CPPMEGA_DSA_SPARSE_MODE:-gather_scatter}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
unset NCCL_NET NCCL_NET_PLUGIN 2>/dev/null || true
export NCCL_NET_PLUGIN=none
export TRITON_CACHE_DIR="${REMOTE_ROOT}/.triton-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "${TRITON_CACHE_DIR}"

# Memory profiling env flag
export CPPMEGA_MEM_PROFILE=1

# Parallelism: same as production but EP=1 and only 5 iters
TP_SIZE=1
PP_SIZE=2
VPP_SIZE=2
EP_SIZE=1
MBS=4
GBS=64
SEQ_LEN=4096
TRAIN_ITERS=5
MTP_DEPTHS=2
NO_ROPE_FUSION=1

echo "[mem_profile] V0 = EP=1 DP=4 PP=2 VPP=2 (all 16 experts/rank, 4x DP, no DeepEP)"
CPPMEGA_DISPATCHER_OVERRIDE=alltoall

# Build workdir
WORKDIR=$(mktemp -d /tmp/cppmega-nam56r-mem-profile.XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Shim: MIMO hooks + fp32-bias + DSA FP8 + MEMORY PROFILER.

When CPPMEGA_MEM_PROFILE=1:
  (A) After model construction: parameter count breakdown by module type
  (B) Per-step memory tracking for first 5 steps
  (C) At exit: full torch.cuda.memory_summary() + parameter breakdown
"""
from __future__ import annotations
import os
import sys
import atexit
import time

_MEM_PROFILE = os.environ.get("CPPMEGA_MEM_PROFILE", "0") == "1"

# =================== (1) deprecate_inference_params compat ===================
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

# =================== (2) MIMO __post_init__ ===================
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

# =================== (3) Mamba3 fp32-bias forward pre-hook ===================
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

# =================== (4) cppmega_mamba3_* __getattr__ fallback ===================
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

# =================== (5) DSA indexer path (bf16-only after FP8 indexer removal) ===================
_dsa_dtype = os.environ.get("CPPMEGA_DSA_INDEXER_DTYPE", "bf16").lower()
print(f"[cppmega_mimo_shim] CPPMEGA_DSA_INDEXER_DTYPE resolves to '{_dsa_dtype}'")
if _dsa_dtype == "fp8":
    raise RuntimeError(
        "CPPMEGA_DSA_INDEXER_DTYPE=fp8 is no longer supported: dsa_fp8_patch.py "
        "and dsa_fp8_indexer.py were deleted on 2026-04-13. Use bf16."
    )

# =================== (6) MEMORY PROFILER ===================
if _MEM_PROFILE:
    import torch
    import collections
    import functools

    _RANK = int(os.environ.get("RANK", "0"))
    _STEP_COUNT = 0
    _MAX_PROFILE_STEPS = 5

    def _mem_gib(b):
        return b / (1024 ** 3)

    def _log_mem(tag):
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        alloc = _mem_gib(torch.cuda.memory_allocated(dev))
        reserved = _mem_gib(torch.cuda.memory_reserved(dev))
        max_alloc = _mem_gib(torch.cuda.max_memory_allocated(dev))
        max_reserved = _mem_gib(torch.cuda.max_memory_reserved(dev))
        print(
            f"[mem_profile] rank={_RANK} {tag}: "
            f"alloc={alloc:.3f} GiB  reserved={reserved:.3f} GiB  "
            f"max_alloc={max_alloc:.3f} GiB  max_reserved={max_reserved:.3f} GiB",
            flush=True,
        )

    def _param_breakdown(model_list, tag="model"):
        """Walk named_parameters and group by module type substring."""
        categories = collections.OrderedDict([
            ("mamba", 0), ("mixer", 0), ("moe", 0), ("router", 0),
            ("attention", 0), ("cross_attn", 0), ("self_attn", 0),
            ("embedding", 0), ("output_layer", 0), ("head", 0),
            ("norm", 0), ("ln", 0), ("layernorm", 0),
            ("linear", 0), ("mlp", 0), ("ffn", 0),
            ("mtp", 0), ("dsa", 0), ("ngram", 0), ("hash", 0),
            ("structure", 0),
        ])
        total_params = 0
        total_bytes = 0
        by_dtype = collections.Counter()
        by_shape_bucket = collections.Counter()
        # More specific grouping by module class
        by_class = collections.Counter()
        param_details = []

        for model in (model_list if isinstance(model_list, (list, tuple)) else [model_list]):
            for name, param in model.named_parameters():
                numel = param.numel()
                nbytes = numel * param.element_size()
                total_params += numel
                total_bytes += nbytes
                by_dtype[str(param.dtype)] += numel

                name_lower = name.lower()
                matched = False
                for cat in categories:
                    if cat in name_lower:
                        categories[cat] += numel
                        matched = True
                        break
                if not matched:
                    categories.setdefault("other", 0)
                    categories["other"] = categories.get("other", 0) + numel

                # Bucket by size
                if numel > 10_000_000:
                    by_shape_bucket["large (>10M)"] += numel
                elif numel > 1_000_000:
                    by_shape_bucket["medium (1M-10M)"] += numel
                elif numel > 100_000:
                    by_shape_bucket["small (100K-1M)"] += numel
                else:
                    by_shape_bucket["tiny (<100K)"] += numel

                # Top-20 largest params
                param_details.append((name, numel, nbytes, str(param.dtype), list(param.shape)))

        # Print summary
        print(f"\n{'='*80}", flush=True)
        print(f"[mem_profile] rank={_RANK} PARAMETER BREAKDOWN ({tag})", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"  Total params:  {total_params:>15,} ({total_params/1e9:.3f}B)", flush=True)
        print(f"  Total bytes:   {total_bytes:>15,} ({_mem_gib(total_bytes):.3f} GiB)", flush=True)
        print(f"", flush=True)

        print(f"  By dtype:", flush=True)
        for dtype, count in sorted(by_dtype.items(), key=lambda x: -x[1]):
            print(f"    {dtype:>20s}: {count:>15,} ({count/1e9:.3f}B)", flush=True)

        print(f"\n  By module category:", flush=True)
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100.0 * count / max(total_params, 1)
                print(f"    {cat:>20s}: {count:>15,} ({count/1e9:.3f}B) {pct:5.1f}%", flush=True)

        print(f"\n  By size bucket:", flush=True)
        for bucket, count in sorted(by_shape_bucket.items(), key=lambda x: -x[1]):
            print(f"    {bucket:>20s}: {count:>15,} ({count/1e9:.3f}B)", flush=True)

        # Top 20 largest parameters
        param_details.sort(key=lambda x: -x[1])
        print(f"\n  Top 20 largest parameters:", flush=True)
        for i, (name, numel, nbytes, dtype, shape) in enumerate(param_details[:20]):
            print(f"    {i+1:>3d}. {name}", flush=True)
            print(f"         {numel:>12,} params  {_mem_gib(nbytes):.4f} GiB  {dtype}  shape={shape}", flush=True)

        # Optimizer memory estimate
        print(f"\n  Optimizer memory estimate (distributed Adam, 8 GPUs):", flush=True)
        local_params = total_params  # this rank's shard
        # With distributed optimizer, each rank holds 1/DP of the optimizer state
        # PP=2 means each rank already has ~half the model params
        # DP = world_size / (TP * PP * EP) = 8 / (1*2*1) = 4
        dp_size = 4
        master_weights = local_params * 4 / dp_size  # fp32 master
        momentum = local_params * 4 / dp_size  # fp32 momentum
        variance = local_params * 4 / dp_size  # fp32 variance
        grads = local_params * 2  # bf16 gradients (not distributed -- each rank computes full grad for its params)
        print(f"    This rank's params (PP shard): {local_params:,} ({_mem_gib(local_params * 2):.3f} GiB bf16)", flush=True)
        print(f"    DP size = {dp_size}", flush=True)
        print(f"    Master weights (fp32, /DP):    {_mem_gib(master_weights):.3f} GiB", flush=True)
        print(f"    Momentum (fp32, /DP):          {_mem_gib(momentum):.3f} GiB", flush=True)
        print(f"    Variance (fp32, /DP):          {_mem_gib(variance):.3f} GiB", flush=True)
        print(f"    Gradients (bf16, full shard):  {_mem_gib(grads):.3f} GiB", flush=True)
        total_opt = master_weights + momentum + variance + grads + local_params * 2
        print(f"    Total model+opt+grad:          {_mem_gib(total_opt):.3f} GiB", flush=True)
        print(f"{'='*80}\n", flush=True)

        return total_params, total_bytes

    # ---- (6a) Hook into model builder to capture model post-construction ----
    try:
        from cppmega.megatron import mamba_builder as _mb
        _orig_builder = _mb.cppmega_mamba_builder

        @functools.wraps(_orig_builder)
        def _profiled_builder(*args, **kwargs):
            _log_mem("before_model_build")
            model = _orig_builder(*args, **kwargs)
            _log_mem("after_model_build")
            if _RANK == 0:
                _param_breakdown(model, tag="single_vp_chunk")
            return model

        _mb.cppmega_mamba_builder = _profiled_builder
        # Also patch the workdir copy
        print("[mem_profile] Model builder memory hook installed", flush=True)
    except Exception as _exc:
        print(f"[mem_profile] Model builder hook failed: {_exc}", file=sys.stderr)

    # ---- (6b) Hook into train_step to log per-step memory ----
    try:
        import megatron.training.training as _mtt
        _orig_train_step = _mtt.train_step

        @functools.wraps(_orig_train_step)
        def _profiled_train_step(*args, **kwargs):
            global _STEP_COUNT
            _STEP_COUNT += 1
            if _STEP_COUNT <= _MAX_PROFILE_STEPS and _RANK == 0:
                _log_mem(f"step_{_STEP_COUNT}_pre")
                torch.cuda.reset_peak_memory_stats()

            result = _orig_train_step(*args, **kwargs)

            if _STEP_COUNT <= _MAX_PROFILE_STEPS and _RANK == 0:
                _log_mem(f"step_{_STEP_COUNT}_post")

            if _STEP_COUNT == _MAX_PROFILE_STEPS and _RANK == 0:
                print(f"\n[mem_profile] Step {_STEP_COUNT} complete -- "
                      f"memory profiling for per-step tracking done.", flush=True)

            return result

        _mtt.train_step = _profiled_train_step
        print("[mem_profile] train_step memory hook installed", flush=True)
    except Exception as _exc:
        print(f"[mem_profile] train_step hook failed: {_exc}", file=sys.stderr)

    # ---- (6c) Hook into setup_model_and_optimizer to capture full model ----
    try:
        import megatron.training.training as _mtt2
        if hasattr(_mtt2, 'setup_model_and_optimizer'):
            _orig_setup = _mtt2.setup_model_and_optimizer

            @functools.wraps(_orig_setup)
            def _profiled_setup(*args, **kwargs):
                _log_mem("before_setup_model_and_optimizer")
                result = _orig_setup(*args, **kwargs)
                _log_mem("after_setup_model_and_optimizer")

                # result is (model_list, optimizer, opt_param_scheduler)
                if _RANK == 0 and isinstance(result, tuple) and len(result) >= 1:
                    model_list = result[0]
                    if isinstance(model_list, (list, tuple)):
                        total_all = 0
                        for i, m in enumerate(model_list):
                            tp, tb = _param_breakdown(m, tag=f"vp_chunk_{i}")
                            total_all += tp
                        print(f"\n[mem_profile] Total params across all VP chunks: "
                              f"{total_all:,} ({total_all/1e9:.3f}B)", flush=True)
                    else:
                        _param_breakdown(model_list, tag="full_model")

                    # Print memory after optimizer is built
                    _log_mem("after_optimizer_construction")

                    # Detailed CUDA allocator stats
                    print(f"\n[mem_profile] CUDA memory summary after model+optimizer construction:", flush=True)
                    print(torch.cuda.memory_summary(abbreviated=False), flush=True)

                return result

            _mtt2.setup_model_and_optimizer = _profiled_setup
            print("[mem_profile] setup_model_and_optimizer hook installed", flush=True)
    except Exception as _exc:
        print(f"[mem_profile] setup_model_and_optimizer hook failed: {_exc}", file=sys.stderr)

    # ---- (6d) atexit: full summary ----
    def _cppmega_full_mem_report():
        try:
            if not torch.cuda.is_available():
                return
            rank = int(os.environ.get("RANK", "0"))
            dev = torch.cuda.current_device()
            print(f"\n{'='*80}", flush=True)
            print(f"[mem_profile] EXIT REPORT rank={rank}", flush=True)
            print(f"{'='*80}", flush=True)

            peak_alloc = _mem_gib(torch.cuda.max_memory_allocated(dev))
            peak_reserved = _mem_gib(torch.cuda.max_memory_reserved(dev))
            cur_alloc = _mem_gib(torch.cuda.memory_allocated(dev))
            cur_reserved = _mem_gib(torch.cuda.memory_reserved(dev))

            print(f"  Current allocated:  {cur_alloc:.3f} GiB", flush=True)
            print(f"  Current reserved:   {cur_reserved:.3f} GiB", flush=True)
            print(f"  Peak allocated:     {peak_alloc:.3f} GiB", flush=True)
            print(f"  Peak reserved:      {peak_reserved:.3f} GiB", flush=True)

            print(f"\n  Full CUDA memory summary:", flush=True)
            print(torch.cuda.memory_summary(abbreviated=False), flush=True)

            # Memory snapshot if available
            try:
                snapshot = torch.cuda.memory_snapshot()
                if snapshot:
                    total_active = sum(b['total_size'] for b in snapshot if b.get('active_size', 0) > 0)
                    total_inactive = sum(b['total_size'] for b in snapshot) - total_active
                    n_blocks = len(snapshot)
                    print(f"\n  Memory snapshot: {n_blocks} blocks", flush=True)
                    print(f"    Active blocks total: {_mem_gib(total_active):.3f} GiB", flush=True)
                    print(f"    Inactive total:      {_mem_gib(total_inactive):.3f} GiB", flush=True)
            except Exception:
                pass

            print(f"{'='*80}\n", flush=True)
        except Exception as _exc:
            print(f"[mem_profile] exit report failed: {_exc}", file=sys.stderr)

    atexit.register(_cppmega_full_mem_report)
    print("[mem_profile] Exit memory report registered", flush=True)

else:
    # Non-profiling: simple peak memory reporter
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

# Build hybrid layer pattern
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

# Native args
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

if [ "${MTP_DEPTHS}" -gt 1 ]; then
  NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/--mtp-num-layers 1/--mtp-num-layers ${MTP_DEPTHS}/")
fi

NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed 's/ *--dsa-indexer-dtype [a-z0-9]*//')

# EP=1: override flex to alltoall
NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/--moe-token-dispatcher-type flex/--moe-token-dispatcher-type alltoall/")
NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/ --moe-router-dtype fp32//")
MOE_EXTRA_FLAGS="--moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0"

echo "NATIVE_ARGS (post-sed): ${NATIVE_ARGS}"

# NO CUDA graphs -- see pure memory without graph overhead
CG_FLAGS=""

# Same selective recompute as production
EXTRA_FLAGS="--recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj --mla-down-proj-fusion"

ROPE_FLAG="--no-rope-fusion"

python -c 'import cppmega, megatron; print("cppmega", cppmega.__version__)'
python - <<PY
import os
dtype = os.environ.get("CPPMEGA_DSA_INDEXER_DTYPE", "bf16").lower()
assert dtype == "bf16", f"expected bf16 live DSA indexer path, got {dtype!r}"
print("live DSA indexer path validated: bf16 only")
PY

# Validate DSA
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

echo "=== MEM PROFILE: NAM56R 7/7 MIMO + DSA 9+4 + EP=1 + NO CUDA GRAPHS ==="
echo "RUN_ID=${RUN_ID} VARIANT=${VARIANT} TP=${TP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE} EP=${EP_SIZE} MBS=${MBS} GBS=${GBS} MTP=${MTP_DEPTHS}"
echo "TRAIN_ITERS=${TRAIN_ITERS}"
echo "CPPMEGA_MEM_PROFILE=${CPPMEGA_MEM_PROFILE}"
echo "CG_FLAGS=(disabled for pure memory profiling)"
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
echo "=== Last 200 lines of ${LOG} ==="
tail -200 "${LOG}"
exit ${EXIT_CODE}
