# Mamba3 Mono CuTe Chunk Wave 2 - 2026-04-30

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Make a bounded H200 Modal path for CuTe DSL, or advance the CUDA WMMA fallback
into a measurable LKQ/apply kernel.

## Official CuTe Sources

Official/NVIDIA sources used for the CuTe image path:

- NVIDIA CUTLASS CuTe DSL quick start:
  <https://docs.nvidia.com/cutlass/4.3.0/media/docs/pythonDSL/quick_start.html>
- NVIDIA CUTLASS CuTe DSL introduction:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html>
- NVIDIA `cutlass.cute` API reference:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html>

The quick start documents `pip install nvidia-cutlass-dsl` as the stable wheel
path.  In Modal I used NVIDIA's package index for the CUTLASS DSL wheel.

## Files Changed

- `scripts/modal_mamba3_mono_chunk_wave2.py`
- `cppmega/megatron/mamba3_mono_chunk_skeleton.py`
- `cppmega/megatron/cuda_ext/mamba3_mono_chunk_skeleton.cpp`
- `cppmega/megatron/cuda_ext/mamba3_mono_chunk_skeleton.cu`
- `tools/probes/mamba3_mono_chunk_smoke.py`
- `docs/status/mamba3_mono_cute_chunk_wave2_2026_04_30.md`

## Modal CuTe Image Path

Added `scripts/modal_mamba3_mono_chunk_wave2.py` with three modes:

- `cute-check`: build a bounded overlay on `ghcr.io/jewelmusicee/cppmega:785c3fd`.
- `cute-gemm`: run the repo's existing CuTe DSL single-GEMM WGMMA H200 smoke.
- `wmma-smoke`: run the CUDA WMMA fallback smoke and benchmark.

Overlay install:

```bash
python -m pip install nvidia-cutlass-dsl==4.4.2 quack-kernels==0.3.10 \
  --extra-index-url https://pypi.nvidia.com
```

`cute-check` command:

```bash
GHCR_TAG=785c3fd modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py --mode cute-check
```

Result:

- `cute_viable`: `true` for imports/API presence.
- `nvidia-cutlass-dsl`: `4.4.2`
- `nvidia-cutlass-dsl-libs-base`: `4.4.2`
- `quack-kernels`: `0.3.10`
- `cuda-python`: `13.2.0`
- `cuda-bindings`: `13.2.0`
- `torch`: `2.13.0.dev20260426+cu132`
- `cutlass.cute` resolved from
  `/usr/local/lib/python3.13/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py`.
- `quack.sm90_utils`, `quack.copy_utils`, and `quack.layout_utils` resolved
  from `/usr/local/lib/python3.13/dist-packages/quack/`.

Package note: pip emitted a preexisting resolver warning that `tilelang
0.1.8+cu132.gitf309d814` wants `z3-solver<4.15.5,>=4.13.0`, while the image
has `z3-solver 4.16.0.0`.  This did not block CuTe/quack imports.

## CuTe Runtime Blocker

`cute-gemm` command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py --mode cute-gemm
```

Result:

- GPU: `NVIDIA H200`
- Package/import path: viable.
- Existing repo CuTe WGMMA kernel: not viable yet.

Exact compile blocker:

```text
DSLRuntimeError: ICE IR Verification Failed
error: "cute.copy(copy_g2s, thr_g2s.partition_S(gA), thr_g2s.partition_D(sA))"
("/opt/cppmega/cppmega/megatron/cute_dsl_mimo/single_gemm_test.py":59:4):
'cute.copy' op '!cute_nvgpu.atom.universal_copy<bf16, 128 b>' cannot
vectorized copy to 8 elements
```

Interpretation: the bounded Modal image can install and import
`nvidia-cutlass-dsl`, `cutlass.cute`, and the `quack` modules supplied by
`quack-kernels`, but the existing CuTe single-GEMM harness needs a 4.4.2
copy-tiler/vectorization update before it can compile and launch.

## WMMA Fallback Advancement

The fallback now has a preallocated output entrypoint:

- Python: `allocate_outputs(...)` and `mono_chunk_skeleton_out(...)`
- Pybind: `mono_chunk_skeleton_out`
- CUDA: validates caller-provided outputs and launches the same chunk kernel
  with optional `zero_outputs`

The smoke probe now accepts:

- `--bench-iters`
- `--bench-warmup`
- `--skip-reference`

The timing mode uses CUDA events around repeated `mono_chunk_skeleton_out`
launches with `zero_outputs=False`, so the reported time excludes Python-side
output allocation and output zero-fill kernels.  The kernel still computes the
existing WMMA `LKQ = K @ Q^T` tile and consumes it in the intra-chunk
`LKQ @ dPhi` apply path.

## H200 WMMA Smoke And Timing

Command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --detach --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode wmma-smoke \
  --shape "--B 1 --S 64 --H 4 --P 64 --bench-iters 100 --bench-warmup 20 --atol 2e-2"
```

The local command runner interrupted non-detached Modal runs after about one
minute, so `--detach` was used for the final run.  The app completed normally.

Result:

- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- GPU spec: `H200`
- Shape: `B=1, S=64, H=4, R=4, N=64, P=64, chunk=16`
- Return code: `0`
- Correctness: pass.
- Max abs diffs:
  - `dv`: `3.6819e-04`
  - `dmimo_v`: `8.9966e-04`
  - `dk_diag`: `5.7446e-10`
  - `dq_diag`: `5.1250e-10`
  - `lkq_checksum`: `3.8743e-06`
- Timing mode: `preallocated_no_zero`
- Warmup/iters: `20 / 100`
- Kernel time: `155.6128 us`
- Estimated per-iteration math counted by the probe:
  - `lkq_wmma`: `8,388,608` flops
  - `lkq_apply`: `3,932,160` flops
  - `k_state`: `8,388,608` flops
- Estimated LKQ-only throughput: `0.0539 TFLOP/s`
- Estimated LKQ plus apply throughput: `0.0792 TFLOP/s`

The low estimated throughput is expected for this fallback shape: it is a
correctness and CTA-dataflow vehicle with scalar consumers and atomics around a
small WMMA tile, not an optimized H200 WGMMA kernel.

## Local Sanity

Local GB10 P=128 check:

```bash
python tools/probes/mamba3_mono_chunk_smoke.py \
  --B 1 --S 16 --H 1 --P 128 --bench-iters 10 --bench-warmup 2 --atol 2e-2
```

Result:

- Correctness: pass.
- Shared storage for P=128: `98304` bytes per CTA.
- Kernel time: `254.7840 us`.

Syntax check:

```bash
python -m py_compile \
  cppmega/megatron/mamba3_mono_chunk_skeleton.py \
  tools/probes/mamba3_mono_chunk_smoke.py \
  scripts/modal_mamba3_mono_chunk_wave2.py
```

## Status

CuTe DSL package viability in Modal: yes, with the Wave 2 overlay.

CuTe DSL kernel viability in Modal: blocked by the existing repo
`single_gemm_test.py` vectorized copy pattern under `nvidia-cutlass-dsl==4.4.2`.

WMMA fallback status: advanced into a measurable LKQ/apply H200 smoke path with
preallocated-output kernel timing.
