# Wave47B MXFP8 Linear Saved Operand Contract

Date: 2026-05-01

## Scope

This wave is limited to the cppmega-side TE Linear/autograd contract. It does
not rebuild or install TransformerEngine.

Changed surfaces:

- `cppmega/recipes/run_profiles.py`
- `scripts/cppmega_fp8_shim.py`
- `tools/probes/te_linear_mxfp8_saved_activation_probe.py`
- `tests/test_run_profiles.py`

## Contract Knob

`PrecisionProfile.mxfp8_linear_kernel_contract` is the typed control for the
Linear backward operand contract. It renders to
`CPPMEGA_TE_MXFP8_LINEAR_KERNEL_CONTRACT` for the shim only as an execution
detail.

Supported values:

- `legacy`: compatibility mode for older TE installations.
- `gemm_ready_v1`: producer-attached GEMM-ready rowwise-transposed operands.
- `compact_direct_v1`: no-materialization acceptance lane. Dense Linear
  backward must consume compact columnwise MXFP8 operands directly through the
  cutlass-native backend.

`compact_direct_v1` requires:

- `mxfp8_bwd_backend='cutlass_native'`
- `mxfp8_compact_columnwise_backward=True`
- `mxfp8_dense_saved_operands=False`
- `mxfp8_transpose_emit_backend='off'`
- `mxfp8_transpose_emit_swizzled=False`
- `mxfp8_transpose_emit_strict=False`

## Acceptance Counters

The compact/direct lane is accepted only when these counters are exactly zero:

- `bf16_fallback_dgrad`
- `bf16_fallback_wgrad`
- `mxfp8_tn_adapter_copy_transpose`
- `mxfp8_tn_sidecar_registry_peak`
- `mxfp8_tn_adapter_saved_transpose_operand`

The probe also checks the stricter diagnostics below are zero:

- `mxfp8_tn_sidecar_registry_peak_bytes`
- `mxfp8_tn_adapter_te_emit_deferred`
- `mxfp8_tn_sidecar_attr_attached`

And it requires direct kernel coverage:

- `mxfp8_cutlass_native_dgrad > 0`
- `mxfp8_cutlass_native_wgrad > 0`

## Merge Status

This patch is merge-ready only if Python compile and run-profile unit tests pass.
The GPU probe additionally depends on the currently installed TransformerEngine
and cppmega extension state; this wave intentionally does not touch the shared
`/home/dave/TransformerEngine` build or install.
