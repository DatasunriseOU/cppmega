# Sparse MLA FP8 / Block-Scaled GEMM Plan (2026-04-25)

## What Was Measured

Local GB10 quarter model, `MBS=4`, `GBS=4`, `seq=4096`, 5 train steps on
real `clang_semantic_4k_v10_train` data.

| SparseMLA FP8 quant path                       | Stable iter 3-5 |  Tok/s | Step-5 train loss | Val loss | Test loss | Max alloc |
| ---------------------------------------------- | --------------: | -----: | ----------------: | -------: | --------: | --------: |
| `CPPMEGA_SPARSE_MLA_FP8_QUANT=local_per_token` |       4371.8 ms | 3747.7 |          7.336141 | 5.958006 |  6.006602 | 25,696 MB |
| `CPPMEGA_SPARSE_MLA_FP8_QUANT=te_tensorwise`   |       4413.8 ms | 3712.0 |          7.383913 | 5.918729 |  5.907243 | 25,696 MB |

The 5-step A/B alone was too noisy: it showed a ~1% apparent slowdown for
`te_tensorwise`, but focused profiling showed that was not coming from
SparseMLA.

Focused `SparseMLA_FP8.forward`, `seq=4096`, `heads=28`, `topk=64`, 8 calls:

| SparseMLA FP8 quant path | CUDA total | Main kernel | Peak alloc |
| --- | ---: | ---: | ---: |
| `local_per_token` | 92.0 ms | 15.9 ms | 1008 MB |
| `te_tensorwise` | 13.4 ms | 13.4 ms | 422 MB |

End-to-end torch profiler on quarter train step 3:

| SparseMLA FP8 quant path | Step wall time | Self CUDA total |
| --- | ---: | ---: |
| `local_per_token` | 6745.5 ms | 4.417 s |
| `te_tensorwise` | 6721.8 ms | 4.386 s |

The old local path spent most of its SparseMLA forward time in
`abs/amax/div/copy_` and allocated much larger temporaries. The runtime is now
hard-wired to TE current/tensorwise scaling.

## Current Code Contract

- SparseMLA FP8 always uses TE current/tensorwise scaling.
- TE `Float8Tensor` input is used zero-copy by viewing TE 2.14 `_data` uint8
  storage as `torch.float8_e4m3fn` and broadcasting the TE tensor scale to the
  TileLang kernel's existing scale shape.
- `CPPMEGA_SPARSE_MLA_FP8_QUANT` is no longer a choice. Only TE aliases
  (`te`, `te_current`, `te_tensorwise`, `tensorwise`) are accepted for old
  launch compatibility; `local_per_token`, `per_token`, and `rowwise` fail
  fast.

## 2026-04-25 QK Runtime Probe Update

`agent/sparse-mla-blockscaled-runtime` adds a QK-only experimental runtime
helper behind `CPPMEGA_SPARSE_MLA_BLOCKSCALED_QK=1`:

- `cppmega.megatron.sparse_mla_ops.sparse_mla_blockscaled_qk_scores`
- `cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_qk.py`
- `tools/probes/sparse_mla_blockscaled_qk_probe.py`

This is not wired into `SparseMLA_FP8.forward` by default.  It accepts
pre-quantized MXFP8 or NVFP4 payloads plus per-block scales and returns raw
SparseMLA QK logits.  The MXFP8 path runs FP8 GEMM per 32-wide K block and
applies block scales inside the K-block loop, avoiding BF16 Q/K
materialization.  The NVFP4 path is a correctness probe that scalar-decodes
packed E2M1 values in TileLang; it is not a native NVFP4 tensor-core path.

Validation and blockers are recorded in
`docs/sparse_mla_blockscaled_qk_runtime_2026_04_25.md`.

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

Dense TE block-scaled status is tracked separately in
`docs/gb10_dense_mxfp8_status_2026_04_25.md`. Important caveat: selected
prequantized forward `general_gemm` calls are fast on GB10, but whole-model
TE `--fp8-recipe mxfp8` training is not currently a drop-in replacement on
this stack. After bypassing TE's sm_12.x recipe guard and padding the
NAM56R no-conv Mamba packed in-projection to a multiple of 32, backward still
fails in TE Linear dgrad with cuBLASLt `CUBLAS_STATUS_NOT_SUPPORTED`.

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
2. Keep SparseMLA on the current TileLang FP8 kernel for now, hard-wired to TE
   tensorwise/current scaling. Do not rewrite SparseMLA around block-scaled
   GEMM until we decide how to represent scale blocks inside sparse/topk
   attention.
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
4. For SparseMLA, use the QK helper as the next kernel-design foothold rather
   than replacing the production path immediately.  Full integration still
   needs online softmax/PV fusion, backward, and direct TE MXFP8/NVFP4 tensor
   extraction.
