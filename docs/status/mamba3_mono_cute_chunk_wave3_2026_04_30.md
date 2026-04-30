# Mamba3 Mono CuTe Chunk Wave 3 - 2026-04-30

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Fix or work around the CuTe DSL WGMMA vectorized-copy blocker from Wave 2 and
get a minimal CuTe WGMMA GEMM or LKQ tile to compile and run on Modal H200.

## Files Changed

- `cppmega/megatron/cute_dsl_mimo/single_gemm_test.py`
- `scripts/modal_mamba3_mono_chunk_wave2.py`
- `docs/status/mamba3_mono_cute_chunk_wave3_2026_04_30.md`

## CuTe Copy Blocker

Wave 2 failed at:

```text
'cute.copy' op '!cute_nvgpu.atom.universal_copy<bf16, 128 b>' cannot
vectorized copy to 8 elements
```

The hand-written `single_gemm_test.py` now uses a 2D row-major tiled-copy
layout for BF16 128-bit global/shared copies:

- thread layout: `(num_threads // threads_per_row, threads_per_row)`
- value layout: `(1, copy_elems)`
- `threads_per_row` capped to a 128-byte cache-line load group

This matches the CuTe DSL/quack SM90 pattern instead of the previous flat
`128 threads x 8 values` tiler. The original IR verifier/vectorized-copy error
is gone.

## Modal CuTe Stack

Command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py --mode cute-check
```

App: `ap-PniYvXbzcvnGrJNsQ7ZxEO`

Result:

- `cute_viable`: `true`
- `nvidia-cutlass-dsl`: `4.4.2`
- `nvidia-cutlass-dsl-libs-base`: `4.4.2`
- `quack-kernels`: `0.3.10`
- `cuda-python`: `13.2.0`
- `cuda-bindings`: `13.2.0`
- `torch`: `2.13.0.dev20260426+cu132`

## Hand-Written CuTe WGMMA Harness

Command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py --mode cute-gemm
```

Final app after the copy/layout patch: `ap-LucswzlcVurbjNi5BqqIBP`

Result:

- GPU: `NVIDIA H200`
- Shape: `M=N=K=64`, BF16 inputs/output, F32 accumulator
- Compile + first launch: `0.68 s`
- Original vectorized-copy ICE: fixed
- Correctness: fail
- Max absolute error: `17.318359`
- Max relative error: `0.615764`
- First printed slice matched reference:
  `C_out[0,:4] == C_ref[0,:4] == [6.0625, 10.5625, 11.8750, -11.7500]`

Current blocker: the manually assembled WGMMA harness now compiles and launches,
but still has a C tile layout/epilogue/data-ordering correctness bug. The next
minimal reproducer is no longer a compiler ICE; it is this incorrect 64x64x64
single-GEMM output under `single_gemm_test.py`.

## Working CuTe WGMMA Workaround

Added `--mode quack-gemm`, which runs a minimal BF16 GEMM through
`quack.gemm_interface.gemm` inside the same Modal overlay. This still uses the
NVIDIA CuTe DSL SM90 path, but delegates the full SM90 WGMMA mainloop and
epilogue layout to `quack-kernels`.

Command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode quack-gemm --m 64 --n 64 --k 64 --iters 1000
```

App: `ap-jp7D3Y8p7o8fj9dQAS0ItB`

Result:

- GPU: `NVIDIA H200`
- Shape: `64x64x64`, BF16
- Correctness: pass
- Max absolute error: `0.0`
- Max relative error: `0.0`
- Compile + first launch: `3.5787 s`
- Timing: `538.6740 us/iter` over `1000` iterations
- Throughput reported by the harness: `0.000973 TFLOP/s`

The timing is launch/persistent-kernel overhead dominated at this tiny shape.
Larger smoke attempts did not return Python diagnostics:

- `256x256x256`, `500` iters, app `ap-MhSJWMZqBAbSt3J4jmT1JH`: Modal
  `RemoteError`, app stopped before JSON return.
- `1024x1024x1024`, `100` iters, app `ap-IRqCp9ih4lwne1QjLwb2UG`: Modal
  `RemoteError`, app stopped before JSON return.

## WMMA Comparison

Wave 2 WMMA fallback baseline remains the usable comparison point:

- Shape: `B=1, S=64, H=4, R=4, N=64, P=64, chunk=16`
- Correctness: pass
- Kernel time: `155.6128 us`
- Estimated LKQ plus apply throughput: `0.0792 TFLOP/s`

A Wave 3 non-detached rerun of the same WMMA smoke, app
`ap-KZ6798Yf7aAebYXzEi5gO6`, hit Modal `RemoteError` before JSON return. The
CuTe 64x64x64 GEMM and WMMA LKQ/apply fallback are not equivalent workloads, so
the direct read is only: CuTe WGMMA now compiles/runs correctly through quack at
minimal GEMM scale; the existing WMMA fallback remains faster on its measured
small LKQ/apply smoke and is still the only validated in-repo chunk path.

## Cleanup

Stopped detached Modal app:

```bash
modal app stop -y ap-l2xcsCmAO8oAbbE0qH0MNP
```

Final cleanup check:

```text
running_or_nonzero_tasks 0
```

## Status

CuTe package/import path: viable.

Original hand-written CuTe copy blocker: fixed/worked around with a 2D
row-major copy tiler.

Hand-written CuTe WGMMA kernel: compiles and launches on H200, but fails
correctness with a C tile/epilogue layout bug.

Minimal CuTe WGMMA GEMM: works on H200 through `quack-kernels` with exact
correctness for 64x64x64 BF16.
