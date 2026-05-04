# MXFP8 vs BF16 Acceptance/Profile Runbook - 2026-05-01

This status note is scoped to profiling harness ownership.  It does not change
model, optimizer, TE, or kernel code.

## Worktree

Run from the isolated worktree created for this measurement pass:

```bash
cd /home/dave/source/cppmega
```

The launcher must be pointed back at this checkout explicitly, because
`scripts/local_gb10_quarter_train.sh` otherwise defaults `ROOT` to
`/home/dave/source/cppmega`.

## Typed Profile Contract

All precision and profiling choices below are rendered through
`cppmega/recipes/run_profiles.py`.  Do not add ad-hoc env-only knobs for this
acceptance loop.

Resolved model/token contract:

```bash
PYTHONPATH=/home/dave/source/cppmega \
  /home/dave/cppmega-venv/bin/python -m cppmega.recipes.run_profiles \
  describe local_gb10_quarter --fp8-recipe mxfp8 --train-iters 20 --mem-profile
```

Expected key values:

- `tokens_per_step=16384`
- `hybrid_layer_pattern=*EME*EME*EMM*/*-/*-`
- MTP depth is 2 through `--mtp-num-layers 2`
- MoE dispatcher is local-safe `alltoall`
- DSA is enabled through `--experimental-attention-variant dsa`

BF16 lane render check:

```bash
PYTHONPATH=/home/dave/source/cppmega \
  /home/dave/cppmega-venv/bin/python -m cppmega.recipes.run_profiles \
  shell local_gb10_quarter --fp8-recipe off --train-iters 20 --mem-profile \
  | rg 'CPPMEGA_FP8|CPPMEGA_PARAM_STORAGE|CPPMEGA_MAIN_GRADS|CPPMEGA_MEM|CPPMEGA_TRAIN_ITERS'
```

Expected BF16 keys:

- `CPPMEGA_FP8_RECIPE=off`
- `CPPMEGA_FP8_FORMAT=hybrid`
- `CPPMEGA_PARAM_STORAGE=bf16`
- `CPPMEGA_MAIN_GRADS_DTYPE=bf16`
- `CPPMEGA_TRAIN_ITERS=20`
- `CPPMEGA_MEM_PROFILE=1`

MXFP8 lane render check:

```bash
PYTHONPATH=/home/dave/source/cppmega \
  /home/dave/cppmega-venv/bin/python -m cppmega.recipes.run_profiles \
  shell local_gb10_quarter --fp8-recipe mxfp8 --train-iters 20 --mem-profile \
  | rg 'CPPMEGA_FP8|CPPMEGA_PARAM_STORAGE|CPPMEGA_TE_MXFP8|CPPMEGA_FLASHINFER|CPPMEGA_MAIN_GRADS|CPPMEGA_MEM|CPPMEGA_TRAIN_ITERS'
```

Expected MXFP8 keys:

- `CPPMEGA_FP8_RECIPE=mxfp8`
- `CPPMEGA_FP8_FORMAT=e4m3`
- `CPPMEGA_PARAM_STORAGE=mxfp8`
- `CPPMEGA_MAIN_GRADS_DTYPE=bf16`
- `CPPMEGA_TE_MXFP8_BWD_BACKEND=te_tn_adapter`
- `CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND=te`
- `CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED=1`
- `CPPMEGA_TE_MXFP8_DENSE_SAVED_OPERANDS=1`
- `CPPMEGA_TE_MXFP8_GROUPED_GEMM_READY_BACKWARD=1`
- `CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0`
- `CPPMEGA_TE_MXFP8_DGRAD_BF16=0`
- `CPPMEGA_TE_MXFP8_WGRAD_BF16=0`

## Machine-Free Preflight

Before collecting numbers, check for compiler/profiler noise:

```bash
ps -ef | rg 'TransformerEngine|pip install --no-build-isolation|ninja|nvcc|cicc|ptxas|cmake' | rg -v rg || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

If the first command prints active TE, CUTLASS, or CUDA extension builds, do not
record acceptance throughput.  Keep the output as the blocker artifact.

After the TE build completes, verify the patched symbols installed into the
venv before running the MXFP8 lane:

```bash
LD_LIBRARY_PATH=/home/dave/TransformerEngine:/home/dave/cppmega-venv/lib/python3.13/site-packages/torch/lib \
  /home/dave/cppmega-venv/bin/python - <<'PY'
import transformer_engine.pytorch  # noqa: F401
import transformer_engine_torch as tex
for name in (
    'mxfp8_quantize_with_rowwise_transpose',
    'mxfp8_split_quantize_with_rowwise_transpose',
    'mxfp8_scaling_transpose_cast',
):
    print(name, hasattr(tex, name))
PY
```

All three symbols must print `True` for the TE-transpose MXFP8 acceptance lane.

## 20-Step Acceptance Commands

Use `flock` so another agent does not overlap the GPU run:

```bash
cd /home/dave/source/cppmega
mkdir -p /home/dave/logs/cppmega_acceptance
```

BF16 baseline:

```bash
RUN_ID=accept_bf16_20_$(date +%Y%m%d_%H%M%S)
ROOT=/home/dave/source/cppmega \
RUN_ID=${RUN_ID} \
LOG=/home/dave/logs/cppmega_acceptance/${RUN_ID}.log \
  flock /tmp/cppmega_gpu_profile.lock \
  bash scripts/local_gb10_quarter_train.sh \
    --fp8-recipe off --train-iters 20 --mem-profile
```

MXFP8 candidate:

```bash
RUN_ID=accept_mxfp8_20_$(date +%Y%m%d_%H%M%S)
ROOT=/home/dave/source/cppmega \
RUN_ID=${RUN_ID} \
LOG=/home/dave/logs/cppmega_acceptance/${RUN_ID}.log \
  flock /tmp/cppmega_gpu_profile.lock \
  bash scripts/local_gb10_quarter_train.sh \
    --fp8-recipe mxfp8 --train-iters 20 --mem-profile
```

Comparison parser:

```bash
PYTHONPATH=/home/dave/source/cppmega \
  /home/dave/cppmega-venv/bin/python tools/profiling/compare_bf16_mxfp8.py \
    --bf16-log /home/dave/logs/cppmega_acceptance/<bf16>.log \
    --mxfp8-log /home/dave/logs/cppmega_acceptance/<mxfp8>.log \
    --hot-step-start 10
```

Acceptance fields to report from the parser:

- hot-step average `ms/iter`
- `tok/sec`
- final train, validation, and test loss if present
- skipped and NaN iteration counts
- setup and peak CUDA allocation
- parameter bytes by storage dtype
- MXFP8 strict counters

Strict MXFP8 acceptance requires:

- `bf16_fallback_dgrad=0`
- `bf16_fallback_wgrad=0`
- `fallback_reasons={}` or absent
- `mxfp8_tn_sidecar_registry_size=0` at process exit
- `mxfp8_tn_sidecar_registry_current_bytes=0` at process exit
- `mxfp8_dense_copy_fallback_*` and grouped copy fallback counters are reported,
  even if non-zero, so the remaining materialization tax is visible.

## Profiler Commands

Collect torch profiler and Nsight Systems in separate runs; CUPTI does not allow
two subscribers in one process.

Torch profiler BF16:

```bash
RUN_ID=accept_bf16_torchprof_$(date +%Y%m%d_%H%M%S)
ROOT=/home/dave/source/cppmega \
RUN_ID=${RUN_ID} \
LOG=/home/dave/logs/cppmega_acceptance/${RUN_ID}.log \
  flock /tmp/cppmega_gpu_profile.lock \
  bash scripts/local_gb10_quarter_train.sh \
    --fp8-recipe off --train-iters 20 --torch-profile --mem-profile
```

Torch profiler MXFP8:

```bash
RUN_ID=accept_mxfp8_torchprof_$(date +%Y%m%d_%H%M%S)
ROOT=/home/dave/source/cppmega \
RUN_ID=${RUN_ID} \
LOG=/home/dave/logs/cppmega_acceptance/${RUN_ID}.log \
  flock /tmp/cppmega_gpu_profile.lock \
  bash scripts/local_gb10_quarter_train.sh \
    --fp8-recipe mxfp8 --train-iters 20 --torch-profile --mem-profile
```

Nsight Systems MXFP8 full capture:

```bash
RUN_ID=accept_mxfp8_nsys_$(date +%Y%m%d_%H%M%S)
ROOT=/home/dave/source/cppmega \
RUN_ID=${RUN_ID} \
LOG=/home/dave/logs/cppmega_acceptance/${RUN_ID}.log \
  flock /tmp/cppmega_gpu_profile.lock \
  bash scripts/local_gb10_quarter_train.sh \
    --fp8-recipe mxfp8 --train-iters 20 \
    --nsys-profile --nsys-capture-mode full --mem-profile
```

Nsight Compute MXFP8 CUDA range, steps 10-12:

```bash
RUN_ID=accept_mxfp8_ncu_$(date +%Y%m%d_%H%M%S)
ROOT=/home/dave/source/cppmega \
RUN_ID=${RUN_ID} \
LOG=/home/dave/logs/cppmega_acceptance/${RUN_ID}.log \
  flock /tmp/cppmega_gpu_profile.lock \
  ncu --target-processes all --set full \
    --export /home/dave/logs/cppmega_acceptance/${RUN_ID}.ncu-rep \
    --force-overwrite \
    bash scripts/local_gb10_quarter_train.sh \
      --fp8-recipe mxfp8 --train-iters 20 --cuda-profile \
      --cuda-profile-step-start 10 --cuda-profile-step-end 12 --mem-profile
```

After each profiler run, summarize with:

```bash
PYTHONPATH=/home/dave/source/cppmega \
  /home/dave/cppmega-venv/bin/python tools/profiling/profile_report.py \
    --log /home/dave/logs/cppmega_acceptance/<run>.log \
    --hot-step-start 10
```

## Highest-Impact Bottleneck Signals To Extract

Prioritize these before changing kernels again:

1. Dense Linear backward materialization: counts and time around
   `mxfp8_dense_copy_fallback_dgrad`, `mxfp8_dense_copy_fallback_wgrad`,
   `mxfp8_tn_adapter_copy_transpose`, and TE transpose emit kernels.
2. Grouped MoE backward materialization: `mxfp8_grouped_gemm_ready_*` hits versus
   grouped transpose copy fallback counters, with per-step grouped GEMM count.
3. Sidecar memory: `mxfp8_tn_sidecar_registry_peak_bytes` and tracked attr peak
   bytes versus max CUDA allocated/reserved.
4. CCE launch count: main plus two MTP heads should use the fused CCE path;
   repeated `_cce_backward_kernel` launches in nsys/torch profiler are a direct
   target.
5. Mamba/M2RNN kernels: `_m2rnn_fwd_kernel`, scan kernels, and any remaining
   full-state materialization should be ranked by self CUDA time and allocation
   pressure.
6. Device-to-device copies from Nsight Systems `cuda_gpu_mem_sum`; if D2D bytes
   track MXFP8 sidecar traffic, the next change is deeper TE/autograd saved
   operand ownership, not another wrapper.

## Current Blocker Snapshot

At 2026-05-01T12:51:37Z, the machine was not free for acceptance/profile
measurement.  Active build noise included:

- `/home/dave/cppmega-venv/bin/python -m pip install --no-build-isolation --no-deps -v /home/dave/TransformerEngine`
- `cmake --build /home/dave/TransformerEngine/build/cmake --verbose --parallel`
- `ninja -v`
- multiple `cicc` and `ptxas` compiler children
- a CUDA extension build from `/home/dave/source/cppmega-wave45B-sm120-compact-producer/cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.cu`

The blocker process snapshot is archived at:

```text
/home/dave/logs/cppmega_acceptance/te_build_blocker_ps_20260501.txt
```

GPU utilization was 0%, but CPU compiler noise makes wall-time and profiler
scheduling numbers invalid for acceptance.

