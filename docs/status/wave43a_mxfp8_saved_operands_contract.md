# Wave43A MXFP8 Saved-Operands Contract

This wave adds typed profile controls for the dense TE Linear MXFP8 backward
operand contract.  The goal is to make BF16 saved activations and full MXFP8
transpose sidecars visible at profile resolution time, then let probes fail fast
when a no-materialization lane accidentally falls back to them.

## Profile Contracts

`mxfp8_linear_kernel_contract=legacy` keeps the old bridge behavior.  It does
not request dense saved MXFP8 operands and it does not enable compact-columnwise
direct backward.

`mxfp8_linear_kernel_contract=gemm_ready_v1` is the merge-ready default.  It
asks TE producer hooks to save GEMM-ready one-shot MXFP8 operands for Linear
backward so the deprecated copy-transpose bridge is avoided when the installed
TE exposes the saved-operand helpers.  This still may materialize GEMM-ready
MXFP8 saved operands; it is a compatibility contract, not the final compact
direct mainloop.

`mxfp8_linear_kernel_contract=compact_direct_v1` is an experimental acceptance
lane for the no-materialization backend.  It resolves to `cutlass_native`, turns
off TE transpose emit, disables dense saved operands, enables compact-columnwise
backward, and disallows BF16 dgrad/wgrad fallback.  The shim rejects saved
rowwise-transpose operands and sidecar registry attachment under this contract.

`mxfp8_grouped_quantize_producer=multi_output` is a typed request for a fused
multi-output split-quantize producer.  If the installed TE has no matching C++
extension, cppmega records a fallback counter and preserves the single-output
path.  This keeps the profile API stable while the TE-side fused producer is
still optional.

## Counters

For `compact_direct_v1`, the focused probe expects these counters to stay zero:

- `mxfp8_tn_adapter_saved_transpose_operand`
- `mxfp8_tn_adapter_copy_transpose`
- `mxfp8_tn_adapter_missing_sidecar_copy`
- `mxfp8_tn_adapter_te_emit`
- `mxfp8_tn_adapter_te_emit_deferred`
- `mxfp8_tn_sidecar_attr_attached`
- `mxfp8_tn_sidecar_registry_peak`

New grouped producer counters:

- `mxfp8_grouped_quantize_multi_output`: TE exposed and cppmega used a fused
  multi-output quantize producer.
- `mxfp8_grouped_quantize_multi_output_fallback`: typed profile requested the
  fused producer, but the installed TE lacked it or the call failed, so cppmega
  fell back to the compatible single-output path.

## Validation Scope

This worktree does not rebuild or install `/home/dave/TransformerEngine`.  The
cppmega changes are therefore limited to typed profile resolution, shim routing,
counter/fail-fast behavior, and focused probes/tests that can run against the
currently installed TE.  The compact-direct kernel performance work remains
experimental until the direct compact-columnwise mainloop beats the
GEMM-ready/materialized TN path on end-to-end 100-step runs.
