# MXFP8 Wgrad Streaming Stock Probe

Date: 2026-04-29

## Scope

This note records the second optimization wave for GB10 SM121 MXFP8 wgrad:

```text
dy compact-columnwise TE MXFP8 + saved x.T rowwise MXFP8
  -> wgrad = dy.T @ x
```

The goal was to test whether the fast stock CUTLASS swizzled-scale GEMM can be
used without materializing full `dy.T` payload/scale sidecars for every wgrad.

## Integrated

- Added an opt-in early B-TMA direct backend:
  `wgrad_nt_gemm_x_rowwise_transpose(..., b_tma_early=True)`.
- Added a probe-only streaming stock path:
  `wgrad_nt_gemm_streaming_swizzled_stock(...)`.
- Added tile producer kernels:
  `prepare_wgrad_stock_a_tile` and `prepare_wgrad_stock_b_scale_tile`.
- Added strided stock GEMM pybind:
  `tn_gemm_swizzled_scale_strided`.
- Extended `tools/probes/cutlass_mxfp8_mixed_wgrad_microbench.py` with
  `legacy`, `a_col_smem_scalar`, `a_col_smem_b_tma_early`,
  `te_emit_swizzled_stock`, and `streaming_swizzled_stock`.

The streaming path is not a default. It is a benchmark/probe backend because it
is not bit-exact with the current compact-direct path, though it matches the
stock swizzled GEMM reference.

## Full-Shape Results

Shape:

```text
dy:           [16384, 3584]
x.T:          [3584, 16384]
logical GEMM: [3584, 3584, 16384]
```

Measured on NVIDIA GB10 with CUDA event timing.

| backend | median ms | extra persistent bytes | full sidecar bytes | parity vs direct |
| --- | ---: | ---: | ---: | --- |
| legacy direct | 47.922 | 0 | 0 | exact |
| a_col_smem_scalar | 26.111 | 0 | 0 | exact |
| a_col_smem_b_tma_early | 26.256 | 0 | 0 | exact |
| te_emit_swizzled_stock | 7.148 | 62,390,272 | 62,390,272 | max_abs 2.44e-4, rel_l2 1.13e-5 |
| streaming_swizzled_stock | 7.312 | 18,350,080 | 0 | max_abs 2.44e-4, rel_l2 1.13e-5 |

Interpretation:

- The stock CUTLASS swizzled-scale GEMM is about 3.6x faster than the accepted
  exact compact direct path.
- Streaming stock removes the 62.39 MB full `dy.T`/scale sidecars and keeps only
  18.35 MB bounded scratch for `tile_m=1024`, `tile_n=2048`.
- The remaining gap to full-sidecar stock is small in kernel timing, but the
  path still launches many per-tile producer/GEMM calls.

## Profiler Artifacts

- Nsight Systems:
  `/home/dave/logs/cutlass_mxfp8_streaming_stock_20260429.nsys-rep`
- Nsight Compute:
  `/home/dave/logs/cutlass_mxfp8_streaming_stock_20260429_ncu.ncu-rep`
- Nsight Systems for early B-TMA whole run:
  `/home/dave/logs/cutlass_mxfp8_b_tma_early_whole_20260429.nsys-rep`
- Nsight Compute for early B-TMA:
  `/home/dave/logs/cutlass_mxfp8_b_tma_early_20260429_ncu.ncu-rep`

Streaming stock Nsight Systems kernel summary for three timed iters:

```text
prepare_wgrad_stock_a_tile_kernel: 50.5%, 12 instances, avg 0.906 ms
stock CUTLASS device_kernel:       47.7%, 24 instances, avg 0.428 ms
prepare_wgrad_stock_b_scale_tile:   1.8%, 24 instances, avg 0.016 ms
```

So the next real target is the A-tile producer. The B-scale producer is already
small.

## A-Tile Producer Fix

The first streaming producer wrote payload bytes with a strided warp-store
pattern:

```text
read  dy_colwise[kk, m_start + row]  contiguous over row
write a_tile[row, kk]                stride K over row
```

It has been replaced with a 32x32 shared-memory tiled transpose. Reads from TE
compact-columnwise `dy` and writes into rowwise `a_tile` are now both
coalesced.

Updated full-shape result for the default streaming tile `1024x2048`:

| backend | median ms | extra persistent bytes | full sidecar bytes | parity vs direct |
| --- | ---: | ---: | ---: | --- |
| streaming_swizzled_stock before A-producer fix | 7.312 | 18,350,080 | 0 | max_abs 2.44e-4, rel_l2 1.13e-5 |
| streaming_swizzled_stock after A-producer fix | 4.105 | 18,350,080 | 0 | max_abs 2.44e-4, rel_l2 1.13e-5 |

New Nsight Systems kernel summary for three timed iters:

```text
stock CUTLASS device_kernel:       85.7%, 24 instances, avg 0.453 ms
prepare_wgrad_stock_a_tile_kernel: 11.3%, 12 instances, avg 0.120 ms
prepare_wgrad_stock_b_scale_tile:   3.0%, 24 instances, avg 0.016 ms
```

Profiler artifacts:

- `/home/dave/logs/cutlass_mxfp8_streaming_stock_a_tile_tiled_20260429.nsys-rep`
- `/home/dave/logs/cutlass_mxfp8_streaming_stock_a_tile_tiled_20260429_ncu.ncu-rep`

Tile sweep after this change showed `1024x2048` and `1024x3584` in the fastest
group. The default remains `1024x2048` because it keeps scratch lower while
matching the best timings within run-to-run jitter.

## Agent Outcomes

- Stock sidecar path remains the fastest raw GEMM consumer, but full sidecars
  cost about 62 MB for this single wgrad shape.
- Raw compact-scale store into native shared layout was not useful; it was
  roughly as slow as the old compact direct path.
- Early B-TMA is correct and can stay as an opt-in probe, but on the latest
  local run it was within jitter and not better than `a_col_smem_scalar`.
- No donor from CUTLASS/FlashInfer/TE was found that can alias TE compact
  columnwise directly into stock SM120 TN GEMM. The practical fast path remains
  either save-time GEMM-ready emit in TE/autograd or a custom grouped producer
  that feeds stock GEMM without per-call full materialization.

## Next Work

1. Replace `prepare_wgrad_stock_a_tile_kernel` with a fused transpose/scale
   producer that feeds a grouped/persistent stock GEMM schedule.
2. Move GEMM-ready backward operand emission deeper into TE Linear/autograd so
   saved activations are already in the stock-consumable layout.
3. Keep `a_col_smem_scalar` as the exact no-sidecar default until the stock path
   has an accepted numerical contract.
