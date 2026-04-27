# GB10 CUTLASS MXFP8 Integrated Backward Backend

Date: 2026-04-26

## Status

The typed profile setting `--mxfp8-bwd-backend cutlass_native` is wired into
`scripts/cppmega_fp8_shim.py` for MXFP8 Linear backward on GB10. The launcher
renders it into `CPPMEGA_TE_MXFP8_BWD_BACKEND` only as import-time shim
transport. The shim still rewrites unsupported TE `NN`/`NT` backward GEMMs to a
GB10-supported `TN` contract, but the GEMM itself can now run through a cppmega
CUTLASS SM120/SM121 extension instead of TE `general_gemm`.

Real-train integration on the 13-layer local GB10 quarter profile found one
remaining coverage gap: `cutlass_native` initially handled the idealized 2D
probe but failed closed on a real TE wgrad operand whose columnwise payload was
matrix-like with extra leading dimensions.  The shim now flattens those leading
dimensions the same way TE computes MXFP8 compact scales; CUTLASS direct still
needs real-train A/B before becoming the default backend.  The current accepted
MXFP8 real-train route is still the typed profile path:
`bash scripts/local_gb10_quarter_train.sh --fp8-recipe mxfp8
--mxfp8-bwd-backend te_tn_adapter --mxfp8-transpose-emit-backend te
--mxfp8-transpose-emit-swizzled --mxfp8-transpose-emit-strict`.

Update later on 2026-04-26: the CUTLASS native backend no longer requires TE
rowwise-transpose sidecars for MXFP8 backward.  For `NN` dgrad and `NT` wgrad,
the shim now passes the original compact TE columnwise payload/scales into a
manual SM120 mainloop loader that writes the CUTLASS shared-memory payload and
scale layouts directly.

Implemented files:

- `cppmega/megatron/cutlass_mxfp8_gemm.py`
- `cppmega/megatron/cuda_ext/cppmega_sm120_blockscaled_mma_tma_compact_scale.hpp`
- `cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.cpp`
- `cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.cu`
- `scripts/cppmega_fp8_shim.py`
- `scripts/local_gb10_quarter_train.sh`
- `tools/probes/gb10_accepted_path_validation.py`
- `tools/probes/gb10_accepted_path_validation_helpers.py`
- `tests/test_gb10_accepted_path_validation.py`

## Current Contract

The extension exposes native CUTLASS MXFP8 `TN` GEMM:

```text
A[M, K] rowwise MXFP8 payload/scales
B[N, K] rowwise MXFP8 payload/scales
out = A @ B.T, BF16
```

By default (`--cutlass-mxfp8-scale-backend compact`), compact TE rowwise E8M0
scale tensors are read directly by a cppmega fork of the CUTLASS SM120
block-scaled mainloop. A/B payload TMA is still stock CUTLASS, but SFA/SFB TMA
descriptors are removed: producer warp lanes copy compact scales into the
native shared-memory scale layout before issuing the A/B TMA copies. This
removes the separate `prepack_two_rowwise_scale_kernel` launch and the native
scale buffer allocation/fill path.

The old cached native-scale prepack path is still available for A/B profiling
through the typed profile:

```bash
bash scripts/local_gb10_quarter_train.sh \
  --fp8-recipe mxfp8 \
  --mxfp8-bwd-backend cutlass_native \
  --mxfp8-transpose-emit-backend off \
  --cutlass-mxfp8-scale-backend prepack
```

The no-sidecar backward path uses the same collective in `manual_payload_load`
mode:

```text
dgrad NN: dy rowwise + weight columnwise -> logical TN dy[M,N] @ weight.T[K,N]
wgrad NT: dy columnwise + x columnwise -> logical TN dy.T[N,M] @ x.T[K,M]
```

This mode keeps the CUTLASS kernel scaffold and pipeline, but producer warp
lanes copy A/B payload bytes and compact E8M0 scales from original TE storage
into CUTLASS shared memory, then manually complete the pipeline transaction.
This avoids both TE rowwise-transpose sidecars and the `.t().contiguous()`
fallback in the shim.

The minimal GB10 path currently requires `M/N/K` multiples of 128 and BF16
outputs.  Unsupported shapes fail closed under `cutlass_native`; they do not
fall back to BF16 when `CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0`.

## TE-Side Transpose Emit

Local TransformerEngine now has
`MXFP8Quantizer.quantize_rowwise_transpose(..., with_gemm_swizzled_scales=...)`
and the matching `transformer_engine_torch.mxfp8_scaling_transpose_cast` op.
The compact mode emits a rowwise MXFP8 payload for `source.T` plus compact
rowwise E8M0 scales by reusing the source tensor's compact columnwise scales.

The 2026-04-26 update extends that path with direct GEMM-swizzled scale
emission.  When `with_gemm_swizzled_scales=True`, the transpose scale kernel
writes the rowwise-transposed scale tensor directly in TE's MXFP8 GEMM swizzled
layout using the existing `gemm_swizzled_scale_idx()` mapping, including the
same padding zeroing as `tex.swizzle_scales_for_gemm_()`.  This avoids the
adapter-side `.t().contiguous()` copy and also avoids a later per-GEMM TE scale
swizzle for the transposed sidecar.

cppmega wires this for the TE TN adapter backend through typed profile args:

```bash
bash scripts/local_gb10_quarter_train.sh \
  --fp8-recipe mxfp8 \
  --mxfp8-bwd-backend te_tn_adapter \
  --mxfp8-transpose-emit-backend te \
  --mxfp8-transpose-emit-swizzled \
  --mxfp8-transpose-emit-strict
```

For `--mxfp8-bwd-backend cutlass_native`, transpose emission stays off by
default. The CUTLASS direct loader consumes original compact TE columnwise
payloads/scales and therefore does not need a sidecar.

## Validation

Syntax/unit checks:

```bash
python3 -m py_compile \
  scripts/cppmega_fp8_shim.py \
  cppmega/megatron/cutlass_mxfp8_gemm.py \
  tools/probes/gb10_accepted_path_validation_helpers.py \
  tests/test_gb10_accepted_path_validation.py
bash -n scripts/local_gb10_quarter_train.sh
PYTHONPATH=/home/dave/source/cppmega pytest -q tests/test_gb10_accepted_path_validation.py
```

Result: `4 passed`.

Correctness probe:

```bash
PYTHONPATH=/home/dave/source/cppmega \
CPPMEGA_ALLOW_TE_MXFP8_SM12=1 \
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1 \
CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native \
CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND=compact \
CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND=off \
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0 \
TORCH_EXTENSIONS_DIR=/tmp/cppmega_cutlass_mxfp8_ext \
/home/dave/cppmega-venv/bin/python \
  tools/probes/te_blockscaled_backward_probe.py \
  --format mxfp8 --use-shim --m 128 --n 128 --k 128
```

Result:

- native TE `NN`/`NT`: still fail with cuBLASLt no-algorithm, as expected.
- `mxfp8_dgrad_shim_NN_to_TN`: pass, `rel_l2=0.0379494`.
- `mxfp8_wgrad_shim_NT_to_TN`: pass, `rel_l2=0.0378986`.
- shim counters: `mxfp8_cutlass_native_dgrad=1`,
  `mxfp8_cutlass_native_wgrad=1`, BF16 fallback, native passthrough,
  transpose sidecar/copy counters, and sidecar registry peak all 0.

Accepted-path validation:

```bash
PYTHONPATH=/home/dave/source/cppmega \
CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native \
CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND=compact \
CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND=off \
TORCH_EXTENSIONS_DIR=/tmp/cppmega_cutlass_mxfp8_ext \
/home/dave/cppmega-venv/bin/python \
  tools/probes/gb10_accepted_path_validation.py \
  --m 128 --n 128 --k 128 --require-mxfp8-probe --probe-timeout-s 180
```

Result: `status=pass`.

TE module smoke:

```text
te.Linear(128,128,bias=False) under MXFP8BlockScaling
loss=0.0683124661
x_grad_finite=true
w_grad_finite=true
mxfp8_cutlass_native_dgrad=1
mxfp8_cutlass_native_wgrad=1
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
mxfp8_tn_adapter_copy_transpose=0
mxfp8_tn_sidecar_registry_peak=0
```

## Microprofile

Warm in-process timing after extension load, shape `128x128x128`:

```text
dgrad_ms = 0.01710
wgrad_ms = 0.02042
```

`nsys` capture:

```text
/home/dave/logs/cutlass_mxfp8_gb10_integrated_micro_20260426.nsys-rep
/home/dave/logs/cutlass_mxfp8_gb10_integrated_micro_20260426.sqlite
/home/dave/logs/cutlass_mxfp8_gb10_integrated_micro_opt1_20260426.nsys-rep
/home/dave/logs/cutlass_mxfp8_gb10_integrated_micro_opt1_20260426.sqlite
```

Original kernel summary for 100 dgrad + 100 wgrad calls:

```text
65.2%  CUTLASS GemmUniversal blockscaled kernel, 210 calls, avg 4.84 us
18.0%  cppmega prepack_rowwise_scale_kernel, 420 calls, avg 0.67 us
15.6%  torch uint8 fill for native scale buffers, 420 calls, avg 0.58 us
```

After caching scale/workspace buffers, replacing `zeros` with `empty`, and
combining A/B prepack:

```text
85.8%  CUTLASS GemmUniversal blockscaled kernel, 220 calls, avg 4.84 us
12.7%  cppmega prepack_two_rowwise_scale_kernel, 220 calls, avg 0.72 us
0.0%   torch uint8 fill for native scale buffers
```

That isolated the remaining hot overhead as the prepack launch itself.  The
compact-scale path below removes it with a custom SM120 loader/mainloop that
keeps stock A/B TMA but reads TE compact scale tensors directly instead of
asking stock CUTLASS scale TMA to consume a layout it rejects.

Compact-scale mainloop (`CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND=compact`) hot
timing after extension load, shape `128x128x128`:

```text
dgrad_ms = 0.01436
wgrad_ms = 0.01792
BF16 fallback counters = 0
```

`nsys` capture:

```text
/home/dave/logs/cutlass_mxfp8_gb10_compact_scale_micro_20260426.nsys-rep
/home/dave/logs/cutlass_mxfp8_gb10_compact_scale_micro_20260426.sqlite
```

Kernel summary for 120 dgrad + 120 wgrad calls:

```text
99.2%  CUTLASS GemmUniversal blockscaled kernel, 240 calls, avg 9.71 us
0.0%   cppmega prepack_rowwise_scale_kernel
0.0%   cppmega prepack_two_rowwise_scale_kernel
```

The compact-scale kernel is slower internally than the prepacked CUTLASS
mainloop because scale copies now happen inside the producer path, but total
microbench latency is lower for the `128x128x128` GB10 backward probe because
the separate scale-prepack launch is gone.

No-sidecar direct loader (`CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND=off`) hot
timing after extension load, shape `128x128x128`, preallocated BF16 outputs:

```text
dgrad_ms = 0.16552
wgrad_ms = 0.11676
mxfp8_cutlass_native_dgrad = 210
mxfp8_cutlass_native_wgrad = 210
mxfp8_tn_adapter_copy_transpose = 0
mxfp8_tn_sidecar_registry_peak = 0
```

`nsys` capture:

```text
/home/dave/logs/cutlass_mxfp8_gb10_direct_micro_20260426.nsys-rep
/home/dave/logs/cutlass_mxfp8_gb10_direct_micro_20260426.sqlite
```

Kernel summary for 50 dgrad + 50 wgrad profiled calls:

```text
99.9%  CUTLASS GemmUniversal blockscaled kernel, 100 calls, avg 142.77 us
0.0%   cppmega prepack_rowwise_scale_kernel
0.0%   cppmega prepack_two_rowwise_scale_kernel
```

The direct loader is a correctness/coverage path, not the final performance
shape: producer warp lanes currently copy the full 128x128 A/B payload tiles
serially into shared memory.  The next optimization target is replacing that
byte-loop with a coalesced/vectorized loader or a conformant TMA/cp.async path
for TE columnwise-transposed payloads.

Update later on 2026-04-26: direct loader opt2 keeps rowwise operands on stock
A/B TMA when their compact rowwise payload has packed `ld == K`, and changes
columnwise-transpose operands from logical row-major scalar byte loops to
K-major, 16-byte vector global loads.  Compact columnwise scale loads are also
K-block-major so each warp reads contiguous TE scale bytes before scattering
into native SM120 shared-memory scale layout.

Microbench command:

```bash
PYTHONPATH=/home/dave/source/cppmega-wt-sm120-loader \
CPPMEGA_ALLOW_TE_MXFP8_SM12=1 \
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1 \
CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native \
CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND=compact \
CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND=off \
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0 \
TORCH_EXTENSIONS_DIR=/tmp/cppmega_cutlass_mxfp8_ext_loader_opt \
/home/dave/cppmega-venv/bin/python \
  tools/probes/te_blockscaled_backward_probe.py \
  --format mxfp8 --use-shim --m 128 --n 128 --k 128 \
  --microbench-cutlass-direct --microbench-warmup 50 --microbench-iters 1000
```

Hot timing after extension load, shape `128x128x128`, preallocated BF16 outputs:

```text
dgrad_ms = 0.02298
wgrad_ms = 0.03769
mxfp8_cutlass_native_dgrad delta = 1050
mxfp8_cutlass_native_wgrad delta = 1050
bf16_fallback_dgrad/wgrad delta = 0
mxfp8_tn_adapter_copy_transpose delta = 0
mxfp8_tn_sidecar_registry_peak delta = 0
```

Correctness vs BF16 reference remains at the expected MXFP8 quantization error:

```text
dgrad rel_l2 = 0.0379494, max_abs = 1.75
wgrad rel_l2 = 0.0378986, max_abs = 1.9375
```

Compared with the previous direct loader (`0.16552 ms` dgrad, `0.11676 ms`
wgrad), opt2 is about `7.2x` faster for dgrad and `3.1x` faster for wgrad.
It is still slower than the materialized compact rowwise TN path (`0.01436 ms`
dgrad, `0.01792 ms` wgrad) because TE columnwise-transpose payloads still must
be scattered into the CUTLASS native shared-memory operand layout by producer
warp lanes.  Current CUTLASS/CUDA TMA descriptor constraints still reject the
desired no-materialization layouts: compact scale TMA hits the zero/compact
stride descriptor path, and payload TMA rejects the required transpose stride
`(1, N)` with a gmem/smem majorness mismatch.
