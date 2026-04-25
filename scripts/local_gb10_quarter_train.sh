#!/usr/bin/env bash
# Local GB10 smoke/debug training for the 13-layer NAM56R-quarter shape.
#
# This launcher intentionally runs from the current cppmega checkout and does
# not reinstall the editable package. It mirrors the known-good GB10 "full dims"
# debug shape: 13 layers, NAM56R per-layer width, real 4k Megatron indexed data.

set -euo pipefail

ROOT="${ROOT:-/home/dave/source/cppmega}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/home/dave/megatron-lm}"
VENV="${VENV:-/home/dave/cppmega-venv}"
RUN_ID="${RUN_ID:-gb10_quarter_fullwidth_$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-/home/dave/logs/${RUN_ID}.log}"
NVSMI_LOG="${NVSMI_LOG:-${LOG%.log}.nvsmi.log}"

source "${VENV}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/dave/.triton-cache}"
export MEGATRON_BIAS_GELU_IMPL="${MEGATRON_BIAS_GELU_IMPL:-te}"
mkdir -p "$(dirname "${LOG}")" "${TRITON_CACHE_DIR}"

export CPPMEGA_NEM_PATTERN="${CPPMEGA_NEM_PATTERN:-AEMEAEMEAEMR}"
export CPPMEGA_LAYER_DEPTH="${CPPMEGA_LAYER_DEPTH:-13}"
export CPPMEGA_DSA_A_LAYER_RANKS="${CPPMEGA_DSA_A_LAYER_RANKS:-1,2,3}"
export CPPMEGA_NGRAM_HASH_ENABLED="${CPPMEGA_NGRAM_HASH_ENABLED:-1}"
export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-1}"
export CPPMEGA_STRUCTURE_COMPONENTS="${CPPMEGA_STRUCTURE_COMPONENTS:-core}"
export CPPMEGA_MAMBA3_MIMO="${CPPMEGA_MAMBA3_MIMO:-1}"
export CPPMEGA_MAMBA_NUM_GROUPS="${CPPMEGA_MAMBA_NUM_GROUPS:-8}"
export CPPMEGA_MAMBA_RECOMPUTE="${CPPMEGA_MAMBA_RECOMPUTE:-1}"
export CPPMEGA_DSA_SPARSE_MODE="${CPPMEGA_DSA_SPARSE_MODE:-tilelang}"
export CPPMEGA_DSA_INDEXER_LOSS_COEFF="${CPPMEGA_DSA_INDEXER_LOSS_COEFF:-0}"
export CPPMEGA_DSA_SKIP_INDEXER_LOSS="${CPPMEGA_DSA_SKIP_INDEXER_LOSS:-1}"
export CPPMEGA_SEQ_LENGTH="${CPPMEGA_SEQ_LENGTH:-4096}"
export CPPMEGA_MAX_POSITION_EMBEDDINGS="${CPPMEGA_MAX_POSITION_EMBEDDINGS:-4096}"
export CPPMEGA_FP8_RECIPE="${CPPMEGA_FP8_RECIPE:-tensorwise}"
export CPPMEGA_SPARSE_MLA_FP8_QUANT="te_tensorwise"
export CPPMEGA_OPTIMIZER="muon"
export CPPMEGA_MUON_MOMENTUM="${CPPMEGA_MUON_MOMENTUM:-0.95}"
export CPPMEGA_MUON_SCALE_MODE="${CPPMEGA_MUON_SCALE_MODE:-spectral}"
export CPPMEGA_MUON_NUM_NS_STEPS="${CPPMEGA_MUON_NUM_NS_STEPS:-5}"
export CPPMEGA_MUON_TP_MODE="${CPPMEGA_MUON_TP_MODE:-blockwise}"
export CPPMEGA_MUON_SCALAR_OPTIMIZER="${CPPMEGA_MUON_SCALAR_OPTIMIZER:-adam8bit}"
export CPPMEGA_MUON_QUANTIZED_MOMENTUM="${CPPMEGA_MUON_QUANTIZED_MOMENTUM:-1}"
export CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE="${CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE:-int8}"
export CPPMEGA_MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE="${CPPMEGA_MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE:-256}"
export CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER=1
export CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER=1
export CPPMEGA_GRAD_REDUCE_IN_BF16=1
export CPPMEGA_USE_DISTRIBUTED_OPTIMIZER="${CPPMEGA_USE_DISTRIBUTED_OPTIMIZER:-0}"
export CPPMEGA_USE_PRECISION_AWARE_OPTIMIZER="${CPPMEGA_USE_PRECISION_AWARE_OPTIMIZER:-0}"
# Aggressive local-GB10 rule: when precision-aware storage is used, never keep
# a higher-precision choice when the next lower storage option exists.
export CPPMEGA_MAIN_GRADS_DTYPE="bf16"
export CPPMEGA_MAIN_PARAMS_DTYPE="fp16"
export CPPMEGA_EXP_AVG_DTYPE="fp8"
export CPPMEGA_EXP_AVG_SQ_DTYPE="fp8"
export CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER="${CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER:-1}"
export CPPMEGA_MEMORY_DEBUG="${CPPMEGA_MEMORY_DEBUG:-0}"
export CPPMEGA_TORCH_PROFILE="${CPPMEGA_TORCH_PROFILE:-0}"
export CPPMEGA_TORCH_PROFILE_STEPS="${CPPMEGA_TORCH_PROFILE_STEPS:-2}"
export CPPMEGA_TORCH_PROFILE_DIR="${CPPMEGA_TORCH_PROFILE_DIR:-/home/dave/logs/${RUN_ID}_torch_profile}"
export CPPMEGA_NSYS_PROFILE="${CPPMEGA_NSYS_PROFILE:-0}"
export CPPMEGA_NSYS_OUTPUT="${CPPMEGA_NSYS_OUTPUT:-/home/dave/logs/${RUN_ID}_nsys}"
if [[ "${CPPMEGA_MEMORY_DEBUG}" == "1" ]]; then
  export NANOCHAT_MEMORY_DEBUG="${NANOCHAT_MEMORY_DEBUG:-1}"
fi

CPPMEGA_DATA_PATH="${CPPMEGA_DATA_PATH:-1.0 /home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10_train}"
CPPMEGA_TOKENIZER_MODEL="${CPPMEGA_TOKENIZER_MODEL:-/home/dave/cppmega-root/cpp_tokenizer_hf}"

WORKDIR="$(mktemp -d /tmp/cppmega-gb10-quarter.XXXXXX)"
cleanup() {
  rm -rf "${WORKDIR}"
  if [[ -n "${NVSMI_PID:-}" ]]; then
    kill "${NVSMI_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${WORKDIR}/torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

export PYTHONPATH="${WORKDIR}:${ROOT}/scripts:${ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"

python - <<PY
import cppmega
path = cppmega.__file__
assert path.startswith("${ROOT}/"), path
print("[local-quarter] cppmega import:", path)
PY

python -m cppmega.megatron.preflight_smem_check

cp "${MEGATRON_ROOT}/pretrain_mamba.py" "${WORKDIR}/pretrain_mamba.py"

cat > "${WORKDIR}/mamba_builders.py" <<'PY'
from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder
PY

cat > "${WORKDIR}/model_provider.py" <<'PY'
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

cat > "${WORKDIR}/cppmega_local_quarter_shim.py" <<'PY'
"""Runtime patches for local GB10 quarter NAM56R training."""
from __future__ import annotations

import collections
import functools
import os
import sys

import cppmega_fp8_shim  # noqa: F401

from cppmega.megatron.apply_linear_ce_patch import (
    patch_mamba_output_layer_with_linear_ce,
)
from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch

apply_dsa_indexer_fused_patch()
patch_mamba_output_layer_with_linear_ce()
print("[local_quarter_shim] full GB10 training patches installed", file=sys.stderr)

if os.environ.get("CPPMEGA_MEM_PROFILE", "0") == "1":
    import torch

    _PROFILE_STEPS = int(os.environ.get("CPPMEGA_MEM_PROFILE_STEPS", "3"))
    _STEP_COUNT = 0
    _PROFILE_MODELS = None
    _MEMORY_DEBUG = os.environ.get("CPPMEGA_MEMORY_DEBUG", "0") == "1" or bool(
        os.environ.get("NANOCHAT_MEMORY_DEBUG", "")
    )

    if _MEMORY_DEBUG:
        os.environ.setdefault("NANOCHAT_MEMORY_DEBUG", "1")
        try:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            print("[memory_debug] Enabled CUDA memory history recording", flush=True)
        except Exception as exc:
            print(f"[memory_debug] Could not enable memory history: {exc}", flush=True)

    def _gib(nbytes: int | float) -> float:
        return float(nbytes) / (1024**3)

    def _log_cuda_mem(tag: str) -> None:
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        print(
            f"[mem_profile] {tag}: "
            f"alloc={_gib(torch.cuda.memory_allocated(dev)):.3f} GiB "
            f"reserved={_gib(torch.cuda.memory_reserved(dev)):.3f} GiB "
            f"max_alloc={_gib(torch.cuda.max_memory_allocated(dev)):.3f} GiB "
            f"max_reserved={_gib(torch.cuda.max_memory_reserved(dev)):.3f} GiB",
            flush=True,
        )

    def _category(name: str) -> str:
        n = name.lower()
        if "ngram" in n or "hash" in n:
            return "ngram_hash"
        if "structure" in n:
            return "structure"
        if "word_embeddings" in n or "embedding" in n:
            return "embedding"
        if "output_layer" in n:
            return "output_layer"
        if "mtp" in n:
            return "mtp"
        if "shared_experts" in n:
            return "moe_shared_experts"
        if ".mlp.experts." in n or "experts." in n:
            return "moe_experts"
        if "router" in n:
            return "moe_router"
        if "self_attention" in n or "cross_attention" in n:
            return "attention"
        if ".mixer." in n or "mamba" in n:
            return "mamba_mixer"
        if ".mlp." in n:
            return "dense_mlp"
        if "norm" in n or "layernorm" in n:
            return "norm"
        return "other"

    def _param_report(models) -> None:
        if not isinstance(models, (list, tuple)):
            models = [models]
        total_numel = 0
        total_bytes = 0
        by_category: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        by_dtype: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        top: list[tuple[int, int, str, str, tuple[int, ...]]] = []

        for model in models:
            for name, param in model.named_parameters():
                numel = int(param.numel())
                nbytes = int(numel * param.element_size())
                total_numel += numel
                total_bytes += nbytes
                cat = _category(name)
                by_category[cat][0] += numel
                by_category[cat][1] += nbytes
                dtype = str(param.dtype)
                by_dtype[dtype][0] += numel
                by_dtype[dtype][1] += nbytes
                top.append((numel, nbytes, name, dtype, tuple(param.shape)))

        print("\n[mem_profile] PARAMETER BREAKDOWN", flush=True)
        print(
            f"[mem_profile] total_params={total_numel:,} "
            f"param_bytes={_gib(total_bytes):.3f} GiB",
            flush=True,
        )
        print("[mem_profile] by_category:", flush=True)
        for cat, (numel, nbytes) in sorted(by_category.items(), key=lambda x: -x[1][1]):
            print(
                f"[mem_profile]   {cat:18s} {numel:>15,} params "
                f"{_gib(nbytes):>8.3f} GiB",
                flush=True,
            )
        print("[mem_profile] by_dtype:", flush=True)
        for dtype, (numel, nbytes) in sorted(by_dtype.items(), key=lambda x: -x[1][1]):
            print(
                f"[mem_profile]   {dtype:18s} {numel:>15,} elems "
                f"{_gib(nbytes):>8.3f} GiB",
                flush=True,
            )

        top.sort(reverse=True)
        print("[mem_profile] top_parameters:", flush=True)
        for i, (numel, nbytes, name, dtype, shape) in enumerate(top[:20], 1):
            print(
                f"[mem_profile]   {i:02d}. {name} "
                f"{numel:,} params {_gib(nbytes):.3f} GiB {dtype} shape={shape}",
                flush=True,
            )

        dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}
        model_bf16 = total_bytes
        grad_fp32 = total_numel * 4
        main_param_fp32 = total_numel * 4
        adam_m_fp32 = total_numel * 4
        adam_v_fp32 = total_numel * 4
        static_est = model_bf16 + grad_fp32 + main_param_fp32 + adam_m_fp32 + adam_v_fp32
        print("[mem_profile] fp32_adam_reference_estimate:", flush=True)
        print(f"[mem_profile]   model_params_actual  {_gib(model_bf16):8.3f} GiB", flush=True)
        print(f"[mem_profile]   grad_buffer_fp32     {_gib(grad_fp32):8.3f} GiB", flush=True)
        print(f"[mem_profile]   main_params_fp32     {_gib(main_param_fp32):8.3f} GiB", flush=True)
        print(f"[mem_profile]   adam_exp_avg_fp32   {_gib(adam_m_fp32):8.3f} GiB", flush=True)
        print(f"[mem_profile]   adam_exp_sq_fp32    {_gib(adam_v_fp32):8.3f} GiB", flush=True)
        print(f"[mem_profile]   reference_total     {_gib(static_est):8.3f} GiB", flush=True)

        if os.environ.get("CPPMEGA_OPTIMIZER", "") == "muon":
            fallback_numel = 0
            muon_numel = 0
            muon_q_blocks = 0
            q_block_size = int(os.environ.get("CPPMEGA_MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE", "256"))
            for model in models:
                for _name, param in model.named_parameters():
                    numel = int(param.numel())
                    is_fallback = (
                        getattr(param, "is_embedding_or_output_parameter", False)
                        or getattr(param, "is_emerging_optimizer_fallback_parameter", False)
                        or len(param.shape) != 2
                    )
                    if is_fallback:
                        fallback_numel += numel
                    else:
                        muon_numel += numel
                        muon_q_blocks += (numel + q_block_size - 1) // q_block_size

            grad_b = 2 if os.environ.get("CPPMEGA_GRAD_REDUCE_IN_BF16", "0") == "1" else 4
            local_no_contig_grad = (
                os.environ.get("CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER", "0") == "1"
            )
            qmuon_enabled = os.environ.get("CPPMEGA_MUON_QUANTIZED_MOMENTUM", "0") == "1"
            if qmuon_enabled:
                muon_state_bytes = muon_numel + muon_q_blocks * 4
                muon_state_desc = (
                    f"q8_data=1B + fp32_absmax/blk{q_block_size} "
                    f"({_gib(muon_state_bytes):.3f} GiB)"
                )
            else:
                muon_state_b = (
                    2
                    if os.environ.get("CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER", "0") == "1"
                    else 4
                )
                muon_state_bytes = muon_numel * muon_state_b
                muon_state_desc = f"{muon_state_b}B ({_gib(muon_state_bytes):.3f} GiB)"
            fallback_no_master = (
                os.environ.get("CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER", "0")
                == "1"
            )
            scalar_opt = os.environ.get("CPPMEGA_MUON_SCALAR_OPTIMIZER", "adam")
            fallback_state_count = 1 if scalar_opt in ("lion", "lion8bit") else 2
            fallback_state_b = 1 if scalar_opt.endswith("8bit") else (2 if fallback_no_master else 4)
            master_b = 0 if fallback_no_master else 4
            persistent_grad_bytes = 0 if local_no_contig_grad else total_numel * grad_b
            live_grad_bytes = total_numel * grad_b
            setup_floor_est = (
                model_bf16
                + persistent_grad_bytes
                + fallback_numel * master_b
                + muon_state_bytes
                + fallback_numel * fallback_state_count * fallback_state_b
            )
            step_live_est = setup_floor_est + (live_grad_bytes if local_no_contig_grad else 0)
            print("[mem_profile] configured_muon_state_estimate:", flush=True)
            print(
                f"[mem_profile]   muon_params={muon_numel:,} fallback_params={fallback_numel:,}",
                flush=True,
            )
            print(
                f"[mem_profile]   grad={grad_b}B "
                f"{'param.grad transient' if local_no_contig_grad else 'persistent DDP buffer'} "
                f"muon_momentum={muon_state_desc} "
                f"fallback_state={fallback_state_count}x{fallback_state_b}B "
                f"fallback_master={master_b}B",
                flush=True,
            )
            print(f"[mem_profile]   setup_floor_total {_gib(setup_floor_est):8.3f} GiB", flush=True)
            if local_no_contig_grad:
                print(
                    f"[mem_profile]   step_live_with_param_grads {_gib(step_live_est):8.3f} GiB",
                    flush=True,
                )

        if os.environ.get("CPPMEGA_USE_PRECISION_AWARE_OPTIMIZER", "0") == "1":
            main_grads_b = dtype_bytes[os.environ.get("CPPMEGA_MAIN_GRADS_DTYPE", "fp32")]
            main_params_b = dtype_bytes[os.environ.get("CPPMEGA_MAIN_PARAMS_DTYPE", "fp32")]
            exp_avg_b = dtype_bytes[os.environ.get("CPPMEGA_EXP_AVG_DTYPE", "fp32")]
            exp_avg_sq_b = dtype_bytes[os.environ.get("CPPMEGA_EXP_AVG_SQ_DTYPE", "fp32")]
            configured_est = (
                model_bf16
                + total_numel * main_grads_b
                + total_numel * main_params_b
                + total_numel * exp_avg_b
                + total_numel * exp_avg_sq_b
            )
            print("[mem_profile] configured_precision_aware_estimate:", flush=True)
            print(
                f"[mem_profile]   main_grads={main_grads_b}B main_params={main_params_b}B "
                f"exp_avg={exp_avg_b}B exp_avg_sq={exp_avg_sq_b}B",
                flush=True,
            )
            print(f"[mem_profile]   configured_total {_gib(configured_est):8.3f} GiB", flush=True)

    try:
        import megatron.training.training as _training

        _orig_setup = _training.setup_model_and_optimizer

        @functools.wraps(_orig_setup)
        def _profiled_setup(*args, **kwargs):
            global _PROFILE_MODELS
            _log_cuda_mem("before_setup_model_and_optimizer")
            result = _orig_setup(*args, **kwargs)
            _log_cuda_mem("after_setup_model_and_optimizer")
            if isinstance(result, tuple) and result:
                _PROFILE_MODELS = result[0]
                _param_report(result[0])
            return result

        _training.setup_model_and_optimizer = _profiled_setup

        _orig_train_step = _training.train_step

        @functools.wraps(_orig_train_step)
        def _profiled_train_step(*args, **kwargs):
            global _STEP_COUNT
            _STEP_COUNT += 1
            if _STEP_COUNT <= _PROFILE_STEPS:
                _log_cuda_mem(f"step_{_STEP_COUNT}_pre")
                torch.cuda.reset_peak_memory_stats()
            result = _orig_train_step(*args, **kwargs)
            if _STEP_COUNT <= _PROFILE_STEPS:
                _log_cuda_mem(f"step_{_STEP_COUNT}_post")
            if _MEMORY_DEBUG and _STEP_COUNT == 1 and _PROFILE_MODELS is not None:
                from cppmega.megatron.memory_debug import dump_memory_after_first_step

                model = _PROFILE_MODELS[0] if isinstance(_PROFILE_MODELS, (list, tuple)) else _PROFILE_MODELS
                dump_memory_after_first_step(model, step=0)
            return result

        _training.train_step = _profiled_train_step
        print("[mem_profile] local memory profile hooks installed", flush=True)
    except Exception as exc:
        print(f"[mem_profile] hook install failed: {exc}", file=sys.stderr)

if os.environ.get("CPPMEGA_TORCH_PROFILE", "0") == "1":
    import torch

    _TORCH_PROFILE_STEPS = int(os.environ.get("CPPMEGA_TORCH_PROFILE_STEPS", "2"))
    _TORCH_PROFILE_DIR = os.environ.get("CPPMEGA_TORCH_PROFILE_DIR")
    _TORCH_PROFILE_STEP = 0
    os.makedirs(_TORCH_PROFILE_DIR, exist_ok=True)

    try:
        import megatron.training.training as _training_profile

        _orig_profile_train_step = _training_profile.train_step

        @functools.wraps(_orig_profile_train_step)
        def _torch_profiled_train_step(*args, **kwargs):
            global _TORCH_PROFILE_STEP
            _TORCH_PROFILE_STEP += 1
            if _TORCH_PROFILE_STEP > _TORCH_PROFILE_STEPS:
                return _orig_profile_train_step(*args, **kwargs)

            trace_path = os.path.join(
                _TORCH_PROFILE_DIR, f"train_step_{_TORCH_PROFILE_STEP}.json"
            )
            table_path = os.path.join(
                _TORCH_PROFILE_DIR, f"train_step_{_TORCH_PROFILE_STEP}_cuda_table.txt"
            )
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                result = _orig_profile_train_step(*args, **kwargs)
            prof.export_chrome_trace(trace_path)
            with open(table_path, "w", encoding="utf-8") as f:
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=80))
            print(
                f"[torch_profile] step={_TORCH_PROFILE_STEP} trace={trace_path} table={table_path}",
                flush=True,
            )
            return result

        _training_profile.train_step = _torch_profiled_train_step
        print(
            f"[torch_profile] hooks installed dir={_TORCH_PROFILE_DIR} "
            f"steps={_TORCH_PROFILE_STEPS}",
            flush=True,
        )
    except Exception as exc:
        print(f"[torch_profile] hook install failed: {exc}", file=sys.stderr)
PY

sed -i '1i import cppmega_local_quarter_shim  # local GB10 quarter train patches' \
  "${WORKDIR}/pretrain_mamba.py"

HYBRID_LAYER_PATTERN="$(python - <<'PY'
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern

print(build_default_hybrid_layer_pattern(mtp_depths=1))
PY
)"

NATIVE_ARGS="$(python - <<'PY'
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan
import os

plan = build_nam56r_feature_plan(
    pattern=os.environ["CPPMEGA_NEM_PATTERN"],
    depth=int(os.environ["CPPMEGA_LAYER_DEPTH"]),
    mtp_depths=1,
)
bundle = build_nam56r_megatron_native_args(
    plan=plan,
    enable_mla=True,
    enable_mtp=True,
    mtp_mode="hybrid",
    enable_moe=True,
)
print(bundle.to_shell_fragment())
PY
)"

(
  while true; do
    ts="$(date '+%Y-%m-%dT%H:%M:%S')"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits |
      while IFS=, read -r mu mt ug tg; do
        echo "${ts},${mu},${mt},${ug},${tg}"
      done
    sleep 2
  done
) > "${NVSMI_LOG}" 2>&1 &
NVSMI_PID=$!

echo "[local-quarter] log=${LOG}"
echo "[local-quarter] nvsmi=${NVSMI_LOG}"
echo "[local-quarter] data=${CPPMEGA_DATA_PATH}"
echo "[local-quarter] depth=${CPPMEGA_LAYER_DEPTH} hidden=${CPPMEGA_HIDDEN_SIZE:-3584} ffn=${CPPMEGA_FFN_HIDDEN_SIZE:-18944} heads=${CPPMEGA_NUM_ATTN_HEADS:-28}"
echo "[local-quarter] torch_extensions=${TORCH_EXTENSIONS_DIR}"
echo "[local-quarter] fp8_recipe=${CPPMEGA_FP8_RECIPE}"
echo "[local-quarter] sparse_mla_fp8_quant=${CPPMEGA_SPARSE_MLA_FP8_QUANT}"
echo "[local-quarter] optimizer=${CPPMEGA_OPTIMIZER} muon_scalar=${CPPMEGA_MUON_SCALAR_OPTIMIZER}"
echo "[local-quarter] no_master_emerging=${CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER} no_master_fallback=${CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER} grad_reduce_bf16=${CPPMEGA_GRAD_REDUCE_IN_BF16}"
echo "[local-quarter] dist_optimizer=${CPPMEGA_USE_DISTRIBUTED_OPTIMIZER} precision_aware=${CPPMEGA_USE_PRECISION_AWARE_OPTIMIZER}"
echo "[local-quarter] local_ddp_no_contig_grad_buffer=${CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER}"
echo "[local-quarter] torch_profile=${CPPMEGA_TORCH_PROFILE} profile_dir=${CPPMEGA_TORCH_PROFILE_DIR}"
echo "[local-quarter] nsys_profile=${CPPMEGA_NSYS_PROFILE} nsys_output=${CPPMEGA_NSYS_OUTPUT}"

# shellcheck disable=SC2206
DATA_ARGS=(--data-path ${CPPMEGA_DATA_PATH})
OPTIMIZER_ARGS=(--optimizer "${CPPMEGA_OPTIMIZER}")
case "${CPPMEGA_OPTIMIZER}" in
  muon|dist_muon|adaptive_muon)
    OPTIMIZER_ARGS+=(
      --muon-momentum "${CPPMEGA_MUON_MOMENTUM}"
      --muon-scale-mode "${CPPMEGA_MUON_SCALE_MODE}"
      --muon-num-ns-steps "${CPPMEGA_MUON_NUM_NS_STEPS}"
      --muon-tp-mode "${CPPMEGA_MUON_TP_MODE}"
      --muon-scalar-optimizer "${CPPMEGA_MUON_SCALAR_OPTIMIZER}"
    )
    if [[ "${CPPMEGA_MUON_QUANTIZED_MOMENTUM}" == "1" ]]; then
      OPTIMIZER_ARGS+=(
        --muon-quantized-momentum
        --muon-quantized-momentum-dtype "${CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE}"
        --muon-quantized-momentum-block-size "${CPPMEGA_MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE}"
      )
    fi
    ;;
esac
if [[ "${CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--use-bf16-no-master-emerging-optimizer)
fi
if [[ "${CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--use-bf16-no-master-emerging-fallback-optimizer)
fi
if [[ "${CPPMEGA_GRAD_REDUCE_IN_BF16}" == "1" || "${CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--grad-reduce-in-bf16)
fi
if [[ "${CPPMEGA_USE_DISTRIBUTED_OPTIMIZER}" == "1" || "${CPPMEGA_USE_PRECISION_AWARE_OPTIMIZER}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--use-distributed-optimizer)
fi
if [[ "${CPPMEGA_USE_PRECISION_AWARE_OPTIMIZER}" == "1" ]]; then
  OPTIMIZER_ARGS+=(
    --use-precision-aware-optimizer
    --main-grads-dtype "${CPPMEGA_MAIN_GRADS_DTYPE}"
    --main-params-dtype "${CPPMEGA_MAIN_PARAMS_DTYPE}"
    --exp-avg-dtype "${CPPMEGA_EXP_AVG_DTYPE}"
    --exp-avg-sq-dtype "${CPPMEGA_EXP_AVG_SQ_DTYPE}"
  )
fi
if [[ "${CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--local-ddp-disable-contiguous-grad-buffer)
fi

LAUNCH_PREFIX=()
if [[ "${CPPMEGA_NSYS_PROFILE}" == "1" ]]; then
  LAUNCH_PREFIX=(
    nsys profile
    --trace=cuda,nvtx,osrt
    --cuda-memory-usage=true
    --sample=none
    --cpuctxsw=none
    --trace-fork-before-exec=true
    --force-overwrite=true
    --wait=all
    -o "${CPPMEGA_NSYS_OUTPUT}"
  )
fi

"${LAUNCH_PREFIX[@]}" python -m torch.distributed.run --nproc_per_node=1 "${WORKDIR}/pretrain_mamba.py" \
  "${DATA_ARGS[@]}" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${CPPMEGA_TOKENIZER_MODEL}" \
  --vocab-size "${CPPMEGA_VOCAB_SIZE:-65536}" \
  --make-vocab-size-divisible-by 128 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}" \
  --hidden-size "${CPPMEGA_HIDDEN_SIZE:-3584}" \
  --ffn-hidden-size "${CPPMEGA_FFN_HIDDEN_SIZE:-18944}" \
  --num-attention-heads "${CPPMEGA_NUM_ATTN_HEADS:-28}" \
  --seq-length "${CPPMEGA_SEQ_LENGTH}" \
  --max-position-embeddings "${CPPMEGA_MAX_POSITION_EMBEDDINGS}" \
  --micro-batch-size "${CPPMEGA_MICRO_BATCH_SIZE:-4}" \
  --global-batch-size "${CPPMEGA_GLOBAL_BATCH_SIZE:-4}" \
  --train-iters "${CPPMEGA_TRAIN_ITERS:-10}" \
  --eval-interval 50000000 \
  --eval-iters 1 \
  --lr "${CPPMEGA_LR:-1e-4}" \
  --min-lr "${CPPMEGA_MIN_LR:-1e-5}" \
  --lr-decay-style constant \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --bf16 \
  --fp8-format hybrid \
  --fp8-recipe "${CPPMEGA_FP8_RECIPE}" \
  --fp8-amax-history-len 1024 \
  --fp8-amax-compute-algo max \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --use-flash-attn \
  --attention-backend flash \
  --spec cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec \
  --cross-entropy-loss-fusion \
  --cross-entropy-fusion-impl linear \
  --recompute-granularity selective \
  --recompute-modules moe_act mlp mla_up_proj \
  --mla-down-proj-fusion \
  --clip-grad 1.0 \
  "${OPTIMIZER_ARGS[@]}" \
  ${NATIVE_ARGS} \
  --moe-token-dispatcher-type alltoall \
  --save-interval 50000000 \
  --log-interval 1 \
  > "${LOG}" 2>&1

tail -n 120 "${LOG}"
echo "--- nvsmi peak ---"
awk -F, '{ if ($2 + 0 > peak) peak = $2 + 0 } END { print "peak_used_mib=" peak }' "${NVSMI_LOG}"
