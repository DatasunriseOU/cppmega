# Mamba3 Mono CuTe Chunk Wave 1 - 2026-04-30

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Prototype a monolithic chunk-owned kernel shape for Mamba3 `bwd_bwd` instead
of adding another incremental sidecar launch.  The target is one CTA owning
`(batch, head, chunk)` and reusing chunk-local intermediates across the `DV`,
`DMIMO_V`, `DK/DQ`, and scalar-consumer families.

## Local And External Search

Local code reviewed:

- `cppmega/megatron/cute_dsl_mimo/fused_bwd_bwd_sm90_p4.py` - existing CuTe
  DSL 10/11-GEMM bwd_bwd chain and loop-carried `dstates` accumulator.
- `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py` - PyTorch
  reference for all bwd_bwd outputs and the useful dependency ordering.
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_chunk_owner_cuda.py`
  plus Modal wave scripts - prior chunk-owner CUDA sidecars and H200 harness
  conventions.
- `/home/dave/vllm/.deps/cutlass-src/examples/python/CuTeDSL/hopper/dense_gemm.py`
  and `hopper/fmha.py` - official local CUTLASS examples for Hopper TMA,
  WGMMA, pipelines, and warp specialization.
- `/home/dave/vllm/.deps/cutlass-src/examples/52_hopper_gather_scatter_fusion/scatter_epilogue.hpp`
  and `35_gemm_softmax/gemm_with_epilogue_visitor.h` - CUTLASS custom
  epilogue and multi-output/aux-output patterns.

Official/NVIDIA references consulted:

- CUTLASS 3.x design and GEMM model:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x.html>
- CUTLASS 3.0 GEMM API and collective mainloop/epilogue composition:
  <https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/gemm_api_3x.html>
- CuTe DSL docs:
  <https://docs.nvidia.com/cutlass/4.4.2/media/docs/pythonDSL/cute_dsl.html>
- CuTe DSL pipeline APIs, including Hopper TMA async pipelines:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html>
- Hopper GMMA/CuTe MMA atom notes:
  <https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0t_mma_atom.html>
- CUTLASS examples 48/50:
  <https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm>
  and
  <https://github.com/NVIDIA/cutlass/tree/main/examples/50_hopper_gemm_with_epilogue_swizzle>

## Prototype

Added a fallback CUDA/CUTLASS-style skeleton because the current prebuilt
Modal image does not contain CuTe DSL packages.

Files:

- `cppmega/megatron/mamba3_mono_chunk_skeleton.py`
- `cppmega/megatron/cuda_ext/mamba3_mono_chunk_skeleton.cpp`
- `cppmega/megatron/cuda_ext/mamba3_mono_chunk_skeleton.cu`
- `tools/probes/mamba3_mono_chunk_smoke.py`
- `scripts/modal_mamba3_mono_chunk_wave1.py`

Kernel shape:

- Grid: `blockIdx = (chunk, head, batch)`.
- Fixed wave-1 math tile: `chunk=16`, `R=4`, `fcs=64`, `N=64`, `P<=128`.
- Threads: 256 threads per CTA.
- Shared memory:
  - `Q[fcs, N]`
  - `K_T[N, fcs]`
  - `dPhi[fcs, P]`
  - `PsiV[fcs, P]`
  - `dPsi[fcs, P]`
  - `LKQ[fcs, fcs]`
- Tensor-core part: WMMA `16x16x16` tiles compute `LKQ = K @ Q^T` as a
  `64x64` tile in shared/register space.  This is a CUDA WMMA stand-in for the
  CuTe/CUTLASS Hopper WGMMA tile that the next wave should replace it with.
- Multi-output epilogue in the same CTA:
  - `DV`
  - `DMIMO_V`
  - diagonal `DK`
  - diagonal `DQ`
  - `LKQ` checksum for smoke validation

The skeleton is deliberately not a production autograd path.  It uses a
pre-expanded per-head layout:

- `q`, `k`: `(B, S, H, R, 64)` fp16
- `dout`, `v`: `(B, S, H, P)` fp16
- `mimo_v`, `mimo_o`: `(H, R, P)` fp16
- `qk_dot`: `(B, S, H, R, R)` fp32
- `dt`, `trap`: `(B, H, S)` fp32
- `dstates`: `(B, H, 64, P)` fp16

## Compile And Smoke Status

Local stack:

- Local device: NVIDIA GB10, capability `(12, 1)`.
- Local CuTe import check: `cutlass`, `cutlass.cute`, `cuda.bindings.driver`,
  `quack`, and `flash_attn.cute.cute_dsl_utils` are present in
  `/home/dave/cppmega-venv`.
- `nvcc`: CUDA 13.2, `V13.2.78`.

Commands run locally:

```bash
python -m py_compile \
  cppmega/megatron/mamba3_mono_chunk_skeleton.py \
  tools/probes/mamba3_mono_chunk_smoke.py \
  scripts/modal_mamba3_mono_chunk_wave1.py

python tools/probes/mamba3_mono_chunk_smoke.py --compile-only
python tools/probes/mamba3_mono_chunk_smoke.py
python tools/probes/mamba3_mono_chunk_smoke.py --B 1 --S 16 --H 1 --P 128 --atol 2e-2
```

Local default smoke result:

- Shape: `B=1, S=32, H=2, R=4, N=64, P=64, chunk=16`.
- Pass: yes.
- Max abs diffs:
  - `dv`: `2.7257e-04`
  - `dmimo_v`: `4.8138e-04`
  - `dk_diag`: `3.7141e-10`
  - `dq_diag`: `4.0149e-10`
  - `lkq_checksum`: `2.3842e-06`

Local P=128 shared-memory smoke:

- Shape: `B=1, S=16, H=1, P=128`.
- Shared memory: `98304` bytes per CTA.
- Pass: yes.

Modal prebuilt image CuTe check:

```bash
GHCR_TAG=785c3fd modal run --timestamps scripts/modal_mamba3_mono_chunk_wave1.py
```

Result on `ghcr.io/jewelmusicee/cppmega:785c3fd`:

- `cutlass`: missing
- `cutlass.cute`: `ModuleNotFoundError: No module named 'cutlass'`
- `nvidia-cutlass-dsl`: missing
- `quack`: missing
- `cute_viable`: false

Exact blocker: the H200 prebuilt Modal image does not install
`nvidia-cutlass-dsl` or `quack`, so the existing CuTe DSL P4 style cannot run
there without rebuilding the image layer.  The fallback CUDA skeleton does not
depend on those packages.

Modal H200 fallback smoke:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 \
  modal run --timestamps scripts/modal_mamba3_mono_chunk_wave1.py --run-gpu-smoke
```

Result:

- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- GPU spec: `H200`
- Return code: `0`
- Shape: `B=1, S=16, H=1, R=4, N=64, P=64, chunk=16`
- Pass: yes.
- Max abs diffs:
  - `dv`: `2.7545e-04`
  - `dmimo_v`: `3.5092e-04`
  - `dk_diag`: `2.4212e-10`
  - `dq_diag`: `2.8837e-10`
  - `lkq_checksum`: `2.9802e-06`

## CuTe Viability

CuTe DSL is viable in the local development environment and there is already
usable repo code in `cppmega/megatron/cute_dsl_mimo/`.

CuTe DSL is not viable in the current H200 Modal prebuilt image
`ghcr.io/jewelmusicee/cppmega:785c3fd` because `cutlass.cute` and `quack` are
absent.  For this wave, the CUDA WMMA fallback is the working H200 path.

## Next Step

Replace the WMMA fallback tile with a true SM90 WGMMA/CuTe or CUTLASS
collective mainloop:

1. Keep the same CTA ownership and shared-memory contract.
2. Move the `LKQ` and state-consumer math into WGMMA tiles with Hopper shared
   memory layouts.
3. Add the missing full bwd_bwd consumers from `full_bwd_bwd_epilogue.py`,
   especially full `DDA_CS`, `DDA_CS_REV`, `DFACTOR`, `DGAMMA`, `DSSDA`,
   `DDA`, and rotary-angle outputs.
4. Rebuild the H200 Modal image with `nvidia-cutlass-dsl` and `quack` if the
   next wave chooses CuTe DSL instead of a C++ CUTLASS kernel.
