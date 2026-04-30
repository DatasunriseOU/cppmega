# MXFP8 Wave13B TE Linear No-Materialization Status

Status: worktree implementation
Date: 2026-04-30

## Scope

Wave13B moves the local GB10 MXFP8 default from materialized TE-TN backward
operands to compact-columnwise saved operands:

- cppmega worktree:
  `/home/dave/source/cppmega-wave13B-te-linear-no-materialize`
- isolated TE worktree:
  `/home/dave/source/TransformerEngine-wave13B-te-linear-no-materialize`
- shared `/home/dave/TransformerEngine` was not edited or installed globally.

The typed profile knob is
`PrecisionProfile.mxfp8_materialization_policy`. The local MXFP8 profile now
sets `compact_columnwise`, which requires `cutlass_native`, transpose emit off,
compact dense backward on, and grouped direct backward on.

## Shape And Layout Contract

Dense `Linear`, `LayerNormLinear`, and `LayerNormMLP` use the same backward
contract:

- dgrad TE call shape is `general_gemm(weight, dy, layout="NN", grad=True)`.
  The cppmega direct backend computes `dX[M,K] = dy[M,N] @ weight[N,K]` from
  `dy` rowwise MXFP8 payload/scales and original `weight` compact-columnwise
  payload/scales.
- wgrad TE call shape is `general_gemm(x, dy, layout="NT", grad=True)`. The
  backend computes `dW[N,K] = dy.T[N,M] @ x[M,K]` from original compact
  columnwise `x` and original compact columnwise `dy`.
- LayerNormLinear and LayerNormMLP treat `x` as the saved norm output or
  activation input. Compact mode must not take the BF16 norm-output bridge and
  must not emit `x.T` or `dy.T` sidecars.
- GroupedLinear dgrad consumes per-expert `dy` rowwise plus per-expert weight
  compact columnwise. Grouped wgrad consumes per-expert `x` compact columnwise
  plus per-expert `dy` compact columnwise.

For compact columnwise scales, the logical leading dimension is padded to TE's
MXFP8 scale-grid requirements. The backend reads the TE tensors through their
stored payload and scale members; it does not reinterpret a PyTorch metadata
transpose as a physical MXFP8 transpose.

## Acceptance Counters

The acceptance signal for the no-materialization lane is:

```text
mxfp8_cutlass_native_dgrad>0
mxfp8_cutlass_native_wgrad>0
mxfp8_grouped_direct_dgrad>0
mxfp8_grouped_direct_wgrad>0
mxfp8_grouped_direct_miss_dgrad=0
mxfp8_grouped_direct_miss_wgrad=0
mxfp8_grouped_transpose_copy_fallback_dgrad=0
mxfp8_grouped_transpose_copy_fallback_wgrad=0
mxfp8_tn_adapter_te_emit=0
mxfp8_tn_adapter_te_emit_deferred=0
mxfp8_tn_adapter_saved_transpose_operand=0
mxfp8_tn_adapter_copy_transpose=0
mxfp8_tn_adapter_missing_sidecar_copy=0
mxfp8_norm_quantize_sidecar_bridge=0
mxfp8_tn_sidecar_attr_attached=0
mxfp8_tn_sidecar_registry_peak=0
mxfp8_tn_sidecar_registry_peak_bytes=0
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
fallback_reasons={}
```

Sidecar bytes must be zero in both the registry peak and current snapshots.
BF16 fallback is not an accepted solution.

## Performance Status

Prior GB10 evidence shows the direct compact loader is slower than the
materialized/copy route even when it clears sidecars:

- `/home/dave/logs/gb10_mxfp8_dense_compact_native5_20260428_224251.log`:
  dense direct counters were clean, but the one-step run was about 241.6 s and
  still had `mxfp8_norm_quantize_sidecar_bridge=35`.
- `/home/dave/logs/gb10_mxfp8_grouped_direct_smoke9_20260428_183814.log`:
  grouped direct counters were clean, but dense still used copy-transpose.

Wave13B removes the TE/autograd materialization source for Linear,
LayerNormLinear, LayerNormMLP, and GroupedLinear. It does not yet prove the
compact direct CUTLASS loader beats the current TE-TN default.

## 2026-04-30 Validation Snapshot

Locked saved-activation probe:

```text
command: flock /tmp/cppmega_gpu_profile.lock -c '... te_linear_mxfp8_saved_activation_probe.py --backend cutlass_native'
shape: M=128, N=128, K=256
status: pass
saved_bf16_input_count=0
saved_transpose_payload_count=0
saved_input_columnwise_payload_count=3
mxfp8_cutlass_native_dgrad=1
mxfp8_cutlass_native_wgrad=1
mxfp8_tn_adapter_copy_transpose=0
mxfp8_tn_adapter_missing_sidecar_copy=0
mxfp8_norm_quantize_sidecar_bridge=0
mxfp8_tn_sidecar_registry_peak_bytes=0
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
fallback_reasons={}
```

Locked one-step full run before the grouped columnwise-weight follow-up patch:

```text
log: /home/dave/logs/wave13B_no_materialize_1step_20260430_0125.log
iteration 1 elapsed: 156899.5 ms
lm loss: 11.66119
max allocated: 24347.85 MB
mxfp8_cutlass_native_dgrad=34
mxfp8_cutlass_native_wgrad=34
mxfp8_tn_adapter_copy_transpose=0
mxfp8_tn_adapter_missing_sidecar_copy=0
mxfp8_norm_quantize_sidecar_bridge=0
mxfp8_tn_sidecar_registry_peak_bytes=0
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
mxfp8_grouped_direct_dgrad=0
mxfp8_grouped_direct_wgrad=10
mxfp8_grouped_direct_miss_dgrad=10
mxfp8_grouped_transpose_copy_fallback_dgrad=10
mxfp8_tn_adapter_saved_transpose_operand=160
fallback_reasons={'ValueError: weight[0] is missing _columnwise_data': 10}
```

The full run proves dense no-materialization and zero BF16 fallback, but not
full grouped acceptance. A TE follow-up patch now emits compact-columnwise
MXFP8 expert-weight operands when primary grouped weights lack columnwise
storage; the grouped-fix one-step rerun was queued behind other locked GPU work
and was not completed in this session.
