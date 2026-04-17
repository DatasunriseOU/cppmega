#!/usr/bin/env bash
# Memory profiling baseline: NAM56R on old main (e40feed4a)
# EP=1 TP=1 PP=2 VPP=2, NO CUDA graphs, TRAIN_ITERS=5
# Prints per-step memory stats + full memory_summary + param counts at exit.
#
# Designed to run on bench3 (h200_1) inside tmux.
set -euo pipefail

[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"

REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data/cppmega-root}"
REMOTE_VENV="${REMOTE_VENV:-/mnt/data/venv}"
RUN_ID="${RUN_ID:-mem_profile_baseline}"
LOG="${LOG:-${REMOTE_ROOT}/cppmega/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-${REMOTE_ROOT}/cppmega/nam56r_grid/${RUN_ID}_ckpt}"

mkdir -p "$(dirname "${LOG}")" "${CKPT_DIR}"
rm -rf "${CKPT_DIR}"/* || true

# Activate venv
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"

# venv NVIDIA libs
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

# Parallelism: EP=1 baseline topology
TP_SIZE=1
PP_SIZE=2
VPP_SIZE=2
EP_SIZE=1
# MBS=1: DSA gather_scatter with topk=256 OOMs even at MBS=2/4 (136+ GiB peak).
MBS=1
GBS=64
SEQ_LEN=4096
TRAIN_ITERS=5
MTP_DEPTHS=2
NO_ROPE_FUSION=1

# EP=1 -> alltoall dispatcher (flex requires TP*EP > 1)
CPPMEGA_DISPATCHER_OVERRIDE=alltoall

# Build workdir with memory-profiling shim
WORKDIR=$(mktemp -d /tmp/cppmega-mem-profile.XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

cat > "${WORKDIR}/cppmega_mimo_shim.py" <<'PY'
"""Memory profiling shim: MIMO hooks + per-step memory stats + exit summary."""
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
    print(f"[mem_profile_shim] static_context alias skipped: {_exc}", file=sys.stderr)

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
        print("[mem_profile_shim] MIMO patch installed (rank=4, chunk=16)")
    except Exception as _exc:
        print(f"[mem_profile_shim] MIMO patch failed: {_exc}", file=sys.stderr)

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
        print("[mem_profile_shim] Mamba3 fp32-bias forward pre-hook installed")
except Exception as _exc:
    print(f"[mem_profile_shim] Mamba3 fp32-bias hook failed: {_exc}", file=sys.stderr)

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

# (5) DSA indexer path (bf16-only after FP8 indexer removal) + loss_coeff==0 gate
_dsa_dtype = os.environ.get("CPPMEGA_DSA_INDEXER_DTYPE", "bf16").lower()
print(f"[mem_profile_shim] CPPMEGA_DSA_INDEXER_DTYPE resolves to '{_dsa_dtype}'")
if _dsa_dtype == "fp8":
    raise RuntimeError(
        "CPPMEGA_DSA_INDEXER_DTYPE=fp8 is no longer supported: dsa_fp8_patch.py "
        "and dsa_fp8_indexer.py were deleted on 2026-04-13. Use bf16."
    )

# (6) Per-step memory tracking via training log hook
_STEP_COUNT = [0]
_MAX_TRACK_STEPS = 5

def _install_step_memory_hook():
    """Monkey-patch megatron's training log to inject memory stats."""
    import torch
    try:
        import megatron.training.training as _mt
        _orig_log_fn = getattr(_mt, 'training_log', None)
        if _orig_log_fn is None:
            print("[mem_profile_shim] WARNING: cannot find megatron.training.training.training_log", file=sys.stderr)
            return

        def _patched_training_log(*args, **kwargs):
            result = _orig_log_fn(*args, **kwargs)
            _STEP_COUNT[0] += 1
            step = _STEP_COUNT[0]
            if step <= _MAX_TRACK_STEPS:
                rank = int(os.environ.get("RANK", "0"))
                dev = torch.cuda.current_device()
                alloc_gib = torch.cuda.memory_allocated(dev) / (1024**3)
                peak_alloc_gib = torch.cuda.max_memory_allocated(dev) / (1024**3)
                reserved_gib = torch.cuda.memory_reserved(dev) / (1024**3)
                peak_reserved_gib = torch.cuda.max_memory_reserved(dev) / (1024**3)
                print(
                    f"[MEM_STEP] rank={rank} step={step} "
                    f"alloc_gib={alloc_gib:.3f} peak_alloc_gib={peak_alloc_gib:.3f} "
                    f"reserved_gib={reserved_gib:.3f} peak_reserved_gib={peak_reserved_gib:.3f}",
                    flush=True,
                )
            return result

        _mt.training_log = _patched_training_log
        print("[mem_profile_shim] per-step memory hook installed")
    except Exception as _exc:
        print(f"[mem_profile_shim] step memory hook failed: {_exc}", file=sys.stderr)

_install_step_memory_hook()

# (7) At-exit: full memory summary + parameter count by module
def _cppmega_mem_exit_report():
    try:
        import torch
        if not torch.cuda.is_available():
            return
        rank = int(os.environ.get("RANK", "0"))
        dev = torch.cuda.current_device()

        # Peak memory
        peak_alloc = torch.cuda.max_memory_allocated(dev) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(dev) / (1024**3)
        print(f"\n{'='*80}", flush=True)
        print(f"[MEM_EXIT] rank={rank} device={dev} "
              f"peak_alloc_gib={peak_alloc:.3f} peak_reserved_gib={peak_reserved:.3f}",
              flush=True)

        # Full memory summary (only on rank 0 to avoid spam)
        if rank == 0:
            print(f"\n[MEM_SUMMARY] torch.cuda.memory_summary():", flush=True)
            print(torch.cuda.memory_summary(dev), flush=True)

        # Parameter count by module type
        try:
            from megatron.training import get_model
            models = get_model._model if hasattr(get_model, '_model') else None
        except Exception:
            models = None

        # Alternative: scan all parameters on this rank via torch modules
        if rank == 0:
            print(f"\n[PARAM_COUNT] Scanning all CUDA tensors on rank={rank}:", flush=True)
            total_params = 0
            total_bytes = 0
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.nn.Module) and hasattr(obj, '_parameters'):
                        pass  # We'll use the model reference approach below
                except Exception:
                    pass

            # Just print total CUDA memory stats
            stats = torch.cuda.memory_stats(dev)
            print(f"\n[CUDA_STATS] Key memory stats:", flush=True)
            for key in sorted(stats.keys()):
                if any(k in key for k in ['allocated', 'reserved', 'active', 'peak']):
                    val = stats[key]
                    if isinstance(val, (int, float)) and val > 1024*1024:
                        print(f"  {key}: {val / (1024**3):.3f} GiB ({val})", flush=True)
                    elif isinstance(val, (int, float)) and val > 0:
                        print(f"  {key}: {val}", flush=True)

        print(f"{'='*80}\n", flush=True)
    except Exception as _exc:
        import traceback
        print(f"[MEM_EXIT] report failed: {_exc}", file=sys.stderr)
        traceback.print_exc()

import gc
atexit.register(_cppmega_mem_exit_report)

# (8) Parameter count reporter -- hooks into model build
def _install_param_reporter():
    """After model is built, print parameter counts grouped by module type."""
    import torch
    try:
        import megatron.training.training as _mt
        _orig_setup = getattr(_mt, 'setup_model_and_optimizer', None)
        if _orig_setup is None:
            print("[mem_profile_shim] WARNING: cannot find setup_model_and_optimizer", file=sys.stderr)
            return

        def _patched_setup(*args, **kwargs):
            result = _orig_setup(*args, **kwargs)
            rank = int(os.environ.get("RANK", "0"))
            if rank == 0:
                models = result[0] if isinstance(result, tuple) else result
                if not isinstance(models, (list, tuple)):
                    models = [models]
                print(f"\n{'='*80}", flush=True)
                print(f"[PARAM_REPORT] Model parameter breakdown (rank=0):", flush=True)
                grand_total = 0
                grand_bytes = 0
                for mi, model in enumerate(models):
                    print(f"\n  --- Virtual Pipeline Stage {mi} ---", flush=True)
                    module_counts = {}
                    for name, param in model.named_parameters():
                        # Group by top-level module
                        parts = name.split('.')
                        # Find meaningful grouping (e.g., 'layers.0.self_attention')
                        if len(parts) >= 4:
                            group = '.'.join(parts[:4])
                        elif len(parts) >= 2:
                            group = '.'.join(parts[:2])
                        else:
                            group = parts[0]

                        n = param.numel()
                        b = n * param.element_size()
                        if group not in module_counts:
                            module_counts[group] = {'params': 0, 'bytes': 0}
                        module_counts[group]['params'] += n
                        module_counts[group]['bytes'] += b
                        grand_total += n
                        grand_bytes += b

                    # Print sorted by bytes descending
                    for grp, info in sorted(module_counts.items(), key=lambda x: -x[1]['bytes']):
                        print(f"    {grp}: {info['params']:>12,} params  {info['bytes']/(1024**2):>8.1f} MiB", flush=True)

                print(f"\n  TOTAL: {grand_total:,} params  {grand_bytes/(1024**3):.3f} GiB", flush=True)

                # Also print optimizer state size estimate
                # Adam: 2 states per param (m, v) in fp32
                opt_state_bytes = grand_total * 4 * 2  # m + v in fp32
                param_fp32_bytes = grand_total * 4  # master weights in fp32
                print(f"  Optimizer state (Adam m+v fp32): {opt_state_bytes/(1024**3):.3f} GiB", flush=True)
                print(f"  Master weights (fp32 copy): {param_fp32_bytes/(1024**3):.3f} GiB", flush=True)
                print(f"  Total model+opt memory est: {(grand_bytes + opt_state_bytes + param_fp32_bytes)/(1024**3):.3f} GiB", flush=True)
                print(f"{'='*80}\n", flush=True)
            return result

        _mt.setup_model_and_optimizer = _patched_setup
        print("[mem_profile_shim] param reporter hook installed")
    except Exception as _exc:
        print(f"[mem_profile_shim] param reporter hook failed: {_exc}", file=sys.stderr)

_install_param_reporter()
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

# Override --mtp-num-layers for MTP_DEPTHS > 1
if [ "${MTP_DEPTHS}" -gt 1 ]; then
  NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/--mtp-num-layers 1/--mtp-num-layers ${MTP_DEPTHS}/")
fi

# Strip --dsa-indexer-dtype (cppmega recipe arg, not Megatron CLI arg)
NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed 's/ *--dsa-indexer-dtype [a-z0-9]*//')

# EP=1: flex -> alltoall override
NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/--moe-token-dispatcher-type flex/--moe-token-dispatcher-type alltoall/")
NATIVE_ARGS=$(echo "${NATIVE_ARGS}" | sed "s/ --moe-router-dtype fp32//")
MOE_EXTRA_FLAGS="--moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0"
echo "NATIVE_ARGS (alltoall override for EP=1): dispatcher=alltoall"

echo "NATIVE_ARGS (post-sed): ${NATIVE_ARGS}"

# CUDA graphs required -- without them DSA gather_scatter OOMs on first forward.
# The 112k baseline ran WITH CG, so this is the correct comparison config.
CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess --cuda-graph-warmup-steps 3"

# Recompute: same as production run for fair comparison
EXTRA_FLAGS="--recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj --mla-down-proj-fusion"

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
PY

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

echo "=== Memory Profile Baseline: NAM56R old main (e40feed4a) ==="
echo "EP=1 TP=1 PP=2 VPP=2 MBS=4 GBS=64 MTP=2 TRAIN_ITERS=5 NO_CG"
echo "EXTRA_FLAGS=${EXTRA_FLAGS}"
echo "LOG=${LOG}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

python -m torch.distributed.run --nproc_per_node=8 "${WORKDIR}/pretrain_mamba.py" \
  --data-path 1.0 /mnt/data/data/megatron/clang_semantic_4k_v10_train \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /mnt/data/tokenizer/ \
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
echo "=== Last 120 lines of ${LOG} ==="
tail -120 "${LOG}"
exit ${EXIT_CODE}
