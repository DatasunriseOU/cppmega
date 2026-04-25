# Sparse MLA FP8 / Block-Scaled GEMM Plan (2026-04-25)

## What Was Measured

Local GB10 quarter model, `MBS=4`, `GBS=4`, `seq=4096`, 5 train steps on
real `clang_semantic_4k_v10_train` data.

| SparseMLA FP8 quant path                       | Stable iter 3-5 |  Tok/s | Step-5 train loss | Val loss | Test loss | Max alloc |
| ---------------------------------------------- | --------------: | -----: | ----------------: | -------: | --------: | --------: |
| `CPPMEGA_SPARSE_MLA_FP8_QUANT=local_per_token` |       4371.8 ms | 3747.7 |          7.336141 | 5.958006 |  6.006602 | 25,696 MB |
| `CPPMEGA_SPARSE_MLA_FP8_QUANT=te_tensorwise`   |       4413.8 ms | 3712.0 |          7.383913 | 5.918729 |  5.907243 | 25,696 MB |

The 5-step loss A/B does not show a correctness red flag for the aggressive
TE tensorwise path, but it also does not improve steady-state throughput on
GB10. The gain is expected to show only if it removes materialization in a
hot DSA path that is otherwise quantizing BF16 activations again.

## Current Code Contract

- Default remains `local_per_token`: BF16/TE input is converted to plain tensor
  and quantized rowwise for the TileLang SparseMLA FP8 kernel.
- Aggressive path is explicit:
  `CPPMEGA_SPARSE_MLA_FP8_QUANT=te_tensorwise`.
- In aggressive mode, TE `Float8Tensor` input is used zero-copy by viewing TE
  2.14 `_data` uint8 storage as `torch.float8_e4m3fn` and broadcasting the
  TE tensor scale to the TileLang kernel's existing scale shape.

## Runtime Options Checked

### RightNow-Tile

Cloned and installed at `/tmp/RightNow-Tile`. It is a Next.js/TypeScript CUDA
SIMT-to-cuTile transpiler UI. `npm install` and `npm run build` pass, but the
repo does not provide runtime kernels, CUTLASS/CuTe block-scaled GEMM code, or
MXFP8/NVFP4 scale layout support. It is not the right integration point for the
training hot path.

### Transformer Engine 2.14

This is the preferred first path.

Installed TE exports:
- `Float8BlockQuantizer`
- `MXFP8Quantizer`
- `NVFP4Quantizer`
- `generic_gemm`
- `multi_tensor_quantize`
- `swizzle_scales_for_gemm_`
- NVFP4/MXFP8 partial-cast and scale kernels

TE MXFP8 tensors already carry rowwise/columnwise data and scale buffers in the
layout expected by TE/cuBLAS GEMM. That makes TE the lowest-risk way to get
block-scaled GEMM without inventing a second scale layout.

### Torch `_scaled_mm`

Torch nightly exposes `_scaled_mm` and `_scaled_mm_v2`. This is useful as a
small 2D cuBLAS-backed reference/probe, but it is not the main training
integration point because TE already wraps the scale swizzle and tensor classes
we need.

### Quack / CUTLASS CuTe DSL

The venv contains `quack.gemm_blockscaled_interface` and a CUTLASS CuTe DSL
Blackwell persistent dense block-scaled GEMM example. This is the right escape
hatch if TE cannot fuse the exact operation:

- Quack supports SM100 MXFP8 blockscaled GEMM with E8M0 scales.
- The wrapper currently normalizes tensors with `.contiguous()` and repacks
  scales before launch, so it is not zero-copy as-is.
- CuTe DSL gives the strongest fusion control for a custom SparseMLA QK/PV
  kernel, but it is the most expensive path to maintain.

## Decision

1. Use TE MXFP8/NVFP4 tensor + `general_gemm/generic_gemm` first for dense
   MLP/MoE/projection paths where GEMM boundaries already exist.
2. Keep SparseMLA on the current TileLang FP8 kernel for now, with the explicit
   `te_tensorwise` A/B flag. Do not rewrite SparseMLA around block-scaled GEMM
   until we decide how to represent scale blocks inside sparse/topk attention.
3. If SparseMLA needs true block-scaled fusion, use CuTe DSL directly, not
   RightNow-Tile. The fused target is:
   - quantize/load Q/K/V in MXFP8 or NVFP4 block layout,
   - run QK block-scaled MMA,
   - apply sparse mask/topk and online softmax,
   - run PV block-scaled/BF16 MMA,
   - write only BF16 output and minimal LSE state.
4. Quack is a useful reference and possible prototype backend, but not a
   production zero-copy path until the `.contiguous()` and scale repack are
   removed or moved into the producer.

## Next Implementation Steps

1. Add a tiny TE block-scaled GEMM probe in tests/benchmarks:
   quantize two BF16 matrices with `MXFP8Quantizer(rowwise=True,
   columnwise=True)` and call TE `general_gemm`; compare against BF16 matmul.
2. Add an adapter that extracts TE MXFP8 rowwise/columnwise data and scale
   shapes for logging only. This tells us whether the producer tensors can
   flow into GEMM without materialization.
3. For MLP/MoE, prefer TE-native block-scaled linear first. This gets us real
   cuBLAS block-scaled GEMM quickly.
4. For SparseMLA, only move to block/MX scales after a kernel design change:
   current TileLang code assumes one scalar per query token and one scalar per
   KV token. Block scales require summing per-K-block products, not just
   multiplying the finished accumulator by one scalar.
