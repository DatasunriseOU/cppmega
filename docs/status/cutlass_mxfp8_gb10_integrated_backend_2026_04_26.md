# GB10 CUTLASS MXFP8 Integrated Backward Backend

Date: 2026-04-26

## Status

`CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native` is now wired into
`scripts/cppmega_fp8_shim.py` for MXFP8 Linear backward on GB10.  The shim
still rewrites unsupported TE `NN`/`NT` backward GEMMs to a GB10-supported
`TN` contract, but the GEMM itself can now run through a cppmega CUTLASS
SM120/SM121 extension instead of TE `general_gemm`.

Implemented files:

- `cppmega/megatron/cutlass_mxfp8_gemm.py`
- `cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.cpp`
- `cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.cu`
- `scripts/cppmega_fp8_shim.py`
- `scripts/local_gb10_quarter_train.sh`
- `tools/probes/gb10_accepted_path_validation_helpers.py`
- `tests/test_gb10_accepted_path_validation.py`

## Current Contract

The extension exposes native CUTLASS MXFP8 `TN` GEMM:

```text
A[M, K] rowwise MXFP8 payload/scales
B[N, K] rowwise MXFP8 payload/scales
out = A @ B.T, BF16
```

Before GEMM, compact rowwise E8M0 scale tensors are prepacked to CUTLASS'
native SM1xx block-scaled layout on GPU.  The current first integrated backend
does not yet own the deeper custom TMA/mainloop loader, so it still relies on
rowwise-transpose operands emitted by the existing TE transpose-sidecar path.
This gives us a real integrated backend and profiler target while keeping the
next no-materialization work localized inside `cutlass_mxfp8_gemm`.

The minimal GB10 path currently requires `M/N/K` multiples of 128 and BF16
outputs.  Unsupported shapes fail closed under `cutlass_native`; they do not
fall back to BF16 when `CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0`.

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
  `mxfp8_cutlass_native_wgrad=1`, BF16 fallback and native passthrough both 0.

Accepted-path validation:

```bash
PYTHONPATH=/home/dave/source/cppmega \
CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native \
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
```

## Microprofile

Warm in-process timing after extension load, shape `128x128x128`:

```text
dgrad_ms = 0.02658
wgrad_ms = 0.02960
```

`nsys` capture:

```text
/home/dave/logs/cutlass_mxfp8_gb10_integrated_micro_20260426.nsys-rep
/home/dave/logs/cutlass_mxfp8_gb10_integrated_micro_20260426.sqlite
```

Kernel summary for 100 dgrad + 100 wgrad calls:

```text
65.2%  CUTLASS GemmUniversal blockscaled kernel, 210 calls, avg 4.84 us
18.0%  cppmega prepack_rowwise_scale_kernel, 420 calls, avg 0.67 us
15.6%  torch uint8 fill for native scale buffers, 420 calls, avg 0.58 us
```

This confirms the first obvious fusion target: stop allocating/zero-filling
native scale buffers every call, then fuse or cache scale prepack where tensor
lifetime allows.  The deeper target remains a custom SM120 loader/mainloop that
reads TE compact scales directly and removes the transpose-sidecar dependency.
