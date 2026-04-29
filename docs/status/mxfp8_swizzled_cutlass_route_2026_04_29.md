# MXFP8 Swizzled CUTLASS Route Status - 2026-04-29

## What Was Added

The local GB10 launcher now has an opt-in typed profile knob:

```bash
--mxfp8-cutlass-scale-backend swizzled
```

This selects `CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native` and routes
GEMM-ready rowwise MXFP8 operands through the stock SM120 CUTLASS
swizzled-scale TN kernel. It keeps BF16 fallback disabled.

## Why It Is Not Default

The microbench result is promising, but the full model is slower until the
producer emits or caches GEMM-ready swizzled scales:

| Run | Steady Step Time | Peak Memory | Notes |
| --- | ---: | ---: | --- |
| BF16 reference | ~4.89-5.09 s | ~27.2 GiB | current local GB10 baseline |
| MXFP8 TE TN default | ~5.29 s | ~25.8 GiB | no BF16 fallback |
| MXFP8 swizzled CUTLASS opt-in | ~7.05-7.19 s | ~26.3 GiB | no BF16 fallback, too many scale swizzles |

Opt-in run:

```text
/home/dave/logs/wave5_swizzled_e2e_20260429_234247.log
```

Key counters:

```text
mxfp8_cutlass_native_dgrad=204
mxfp8_cutlass_native_wgrad=204
mxfp8_cutlass_native_stock_swizzled=204
mxfp8_cutlass_native_stock_scale_swizzle=408
mxfp8_dense_copy_fallback_dgrad=0
mxfp8_dense_copy_fallback_wgrad=0
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
```

The route removed dense copy fallback, but each dense backward GEMM still pays
scale-layout conversion. That dominates the model and makes it slower than both
BF16 and the TE-TN MXFP8 default.

## Next Required Fix

Do not make this route default until one of these is true:

1. TE/autograd emits GEMM-ready rowwise-transpose operands and swizzled scales
   at the producer, so `mxfp8_cutlass_native_stock_scale_swizzle` is near zero.
2. The swizzled scale cache is owned by the MXFP8 tensor producer and has a
   correct invalidation contract for reused TE scale storage.
3. A grouped MoE path avoids falling back to grouped transpose copies while
   using the same stock swizzled-scale mainloop.

The accepted invariant remains:

```text
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
```
