# CUTLASS FP8 Blockwise Transpose-Avoidance Probe

Date: 2026-04-26

Scope:

- `tools/probes/cutlass_fp8_blockwise_transpose_probe.py`
- `/tmp` sanity builds of CUTLASS example 87b

## Summary

The simple stock CUTLASS SM120/SM121 blockwise/software-scale route is not a
drop-in no-copy path for original TE MXFP8 `columnwise_data`.

Two independent constraints block it:

1. Stock SM120 blockwise builder only supports TN/K-major operand layouts.  For
   B, `cutlass::layout::ColumnMajor` on logical `(N,K)` maps to stride
   `(K,1)`, so it requires materialized physical `[N,K]` payload storage.
   Original TE MXFP8 columnwise payload for a source `[K,N]` needs logical
   transpose indexing `B(n,k) = source[k,n]`, i.e. stride `(1,N)`.
2. TE/MXFP8 scale semantics need K-block 32.  Stock SM120 blockwise requires
   `ScaleGranularityK == TileShape_K`; forcing `TileK=32` compiles in a probe
   but fails math / runtime sanity, while upstream 87b with `TileK=128` passes.

The compact `columnwise_scale_inv` indexing itself is straightforward:
`SFB(n,kb) = columnwise_scale_inv[kb,n]`, implemented as logical SFB
`(N,K/32)` with stride over K-blocks equal to padded `N`.  For MXFP8 E8M0 this
still needs a decode/custom scale-load hook because the stock software-scale
mainloop loads FP32 scales.

## Source Refs

- `include/cutlass/detail/layout.hpp:77` maps B RowMajor to stride `(1,ld)`;
  `include/cutlass/detail/layout.hpp:86` maps B ColumnMajor to stride `(ld,1)`.
- `include/cutlass/gemm/collective/builders/sm1xx_common.inl:117` maps B
  RowMajor to `UMMA::Major::MN` and B ColumnMajor to `UMMA::Major::K`.
- `include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl:149`
  derives A/B UMMA majors, and line `151` asserts only TN layout.
- `include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl:197`
  derives `ScaleGranularityK`; line `201` asserts it must equal `TileShape_K`.
- `include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp:129`
  derives scale granularities; line `136` repeats the `TileShape_K` equality
  assertion.
- `include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp:297`
  constructs B as a logical `(N,K,L)` tensor for TMA.
- `include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp:386`
  builds SFA/SFB tensors from `filter(layout_SF*)`; lines `449-451` tile SFB by
  `n_coord` and K tile.
- `include/cutlass/detail/blockwise_scale_layout.hpp:51` defines the SFA/SFB
  nested layout types; lines `153-179` show runtime SFB shape/stride deduction.
- `examples/87_blackwell_geforce_gemm_blockwise/87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu:127`
  uses `ScaleGranularityM=1,N=128,K=128`.

## Commands Run

Layout-only address check:

```bash
python tools/probes/cutlass_fp8_blockwise_transpose_probe.py \
  --layout-only --m 128 --n 128 --k 128
```

Result: `layout.status = pass`. Desired B no-copy offset equals
`source[k,n]`, and SFB direct compact offset equals `columnwise_scale[kb,n]`.
The same report shows stock SM120 B K-major offset diverges from the original
payload for non-zero `n`/`k`.

CUTLASS extension probe:

```bash
python tools/probes/cutlass_fp8_blockwise_transpose_probe.py \
  --m 128 --n 128 --k 128 --require-gemm
```

Result: compiles and launches on GB10 / sm_121a with CUDA 13.2, but reports
`control_failed`; both original payload and materialized-transpose control are
bad math under the forced `TileK=32, ScaleK=32` stock SM120 instantiation.

Upstream sanity:

```bash
nvcc -std=c++17 -O2 --expt-relaxed-constexpr --expt-extended-lambda \
  -arch=sm_121a \
  -I/home/dave/vllm/.deps/cutlass-src/include \
  -I/home/dave/vllm/.deps/cutlass-src/tools/util/include \
  -I/home/dave/vllm/.deps/cutlass-src/examples/common \
  -I/home/dave/vllm/.deps/cutlass-src/examples/87_blackwell_geforce_gemm_blockwise \
  /home/dave/vllm/.deps/cutlass-src/examples/87_blackwell_geforce_gemm_blockwise/87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu \
  -o /tmp/cutlass_87b_probe
/tmp/cutlass_87b_probe --m=128 --n=128 --k=128 --iterations=0
```

Result: upstream 87b cooperative and pingpong schedules both passed.

Forced K32 sanity:

```bash
# /tmp copy of 87b with Cooperative/Pingpong tile K changed to 32,
# ScaleGranularityK changed to 32, and one variant also ScaleGranularityN=1.
/tmp/cutlass_87b_k32_g1_probe --m=128 --n=128 --k=128 --iterations=0
/tmp/cutlass_87b_k32_n128_probe --m=128 --n=128 --k=128 --iterations=0
```

Results:

- `TileK=32, ScaleN=1, ScaleK=32`: verification failed.
- `TileK=32, ScaleN=128, ScaleK=32`: cooperative run returned CUTLASS internal
  error.

## Recommendation

Do not route TE MXFP8 backward through the stock CUTLASS SM120 blockwise
software-scale builder as-is.

Viable next steps:

1. If staying with CUTLASS, fork/customize the SM120 blockwise mainloop:
   support a B MN-major TMA/layout path for `B(n,k)=source[k,n]`, and replace
   FP32 scale loads with compact E8M0 decode or a predecoded FP32 scale view.
2. Alternatively, use SM100/SM90-style blockwise mainloop ideas as reference,
   because those paths allow `ScaleGranularityK` to divide `TileShape_K`; SM120
   stock path does not.
3. For shortest production path, keep the existing transpose-emit/copy adapter
   unless a custom CUTLASS mainloop is worth owning.  The scale indexing piece
   is easy; the payload major requirement and K32 behavior are the real blockers.
