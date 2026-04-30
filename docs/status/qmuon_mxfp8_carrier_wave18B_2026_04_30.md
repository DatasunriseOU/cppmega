Status: active
Canonical: docs/quantized_muon_momentum.md
Date: 2026-04-30
Scope: Wave18B qMuon MXFP8 carrier probe, audit evidence, and staged optimizer integration plan.

# Wave18B qMuon MXFP8 Carrier

## Decision

Stage this as a probe and audit scaffold, not as full optimizer acceptance.

The second locked microprobe is the first useful signal for a non-BF16 qMuon
carrier:

```text
/home/dave/logs/wave18B_qmuon_mxfp8_carrier_probe_20260430_002.log
device=NVIDIA GB10 capability=(12, 1)
config: shape=(4096, 4096) numel=16,777,216 grad_dtype=torch.float16 beta=0.95 warmup=10 iters=30
carrier: data_dtype=torch.uint8 data_shape=(4096, 4096) scale_dtype=torch.uint8 scale_shape=(4096, 128) inv_norm=1.24624046e-03
correctness: carrier_gram_rel_l2=0.00172372 carrier_finite=True gram_finite=True
perf_ms: q_update_emit_mxfp8=0.8691 mxfp8_gram=1.2328 q_update_emit_plus_mxfp8_gram=2.0756 bf16_update_norm_gram=3.8241 speedup_vs_bf16=1.842x
```

The carrier audit probe also shows the qMuon carrier call itself did not
materialize BF16 qMuon grad or NS tensors:

```text
/home/dave/logs/wave18B_qmuon_mxfp8_carrier_audit_20260430_001.log
[cppmega_muon_dtype_audit] bf16_owned_path_observed=0 qmuon_absmax_dtype_float32_elems=64 qmuon_absmax_dtype_float32_tensors=1 qmuon_carrier_inv_norm_dtype_float32_elems=1 qmuon_carrier_inv_norm_dtype_float32_tensors=1 qmuon_carrier_rowwise_data_dtype_uint8_elems=16384 qmuon_carrier_rowwise_data_dtype_uint8_tensors=1 qmuon_carrier_rowwise_scale_dtype_uint8_elems=512 qmuon_carrier_rowwise_scale_dtype_uint8_tensors=1 qmuon_grad_dtype_float16_elems=16384 qmuon_grad_dtype_float16_tensors=1 qmuon_grad_tensors=1 qmuon_mxfp8_carrier_update_calls=1 qmuon_state_dtype_int8_elems=16384 qmuon_state_dtype_int8_tensors=1
carrier_data_dtype=torch.uint8 carrier_scale_dtype=torch.uint8 grad_dtype=torch.float16 inv_norm_dtype=torch.float32
```

## What Exists

Implementation commit: `ab798ed` (`Prototype qMuon MXFP8 carrier probe`).

Added pieces:

- `QuantizedMuonMxfp8Carrier`: rowwise uint8 E4M3 payload plus uint8 E8M0
  inverse scales and one FP32 inverse Frobenius norm scalar.
- `empty_mxfp8_carrier_like` and `dequantize_mxfp8_carrier` for tests and
  probe reference checks.
- `quantized_muon_momentum_update_mxfp8_carrier_`: single-tensor CUDA path
  that updates int8 qMuon state and emits the normalized carrier without
  overwriting the grad tensor with BF16 scratch.
- `tools/probes/qmuon_mxfp8_carrier_ns_probe.py`: one-tensor carrier emission
  plus first Newton-Schulz Gram GEMM probe using the existing CUTLASS rowwise
  MXFP8 TN GEMM helper.
- `cppmega.megatron.muon_dtype_audit`: runtime dtype counters for qMuon state,
  qMuon grad inputs, MXFP8 carrier payload/scales, and Megatron NS helpers.
- Run-profile and local launcher knobs:
  `--muon-ns-carrier {bf16,mxfp8_probe}` and `--muon-dtype-audit`.

Focused tests:

```text
flock /tmp/cppmega_gpu_profile.lock python -m pytest tests/test_muon_dtype_audit.py tests/test_run_profiles.py tests/test_quantized_muon_momentum.py -q
42 passed, 19 warnings
```

## Current Limits

This is not yet a full no-BF16 training path.

- The probe covers one 2D tensor, not the full `TensorParallelMuon` grouped
  optimizer path, QKV split path, tensor-parallel norm all-reduce, or real loss
  trajectory.
- No locked 6-step training run has used this carrier in model training.
- The existing `cppmega.megatron.cutlass_mxfp8_gemm` wrapper still requires
  BF16 output tensors for MXFP8 GEMMs. The carrier storage is non-BF16, but the
  first Gram output in the probe is still BF16 through that wrapper.
- The probe validates the carrier plus `X @ X.T` building block. It does not
  implement all Newton-Schulz intermediates (`A`, `A @ A`, `B @ X`) in MXFP8 or
  in a strict non-BF16 output dtype.
- Shape support is currently limited to CUTLASS-compatible dimensions; the
  probe rejects rows or columns that are not multiples of 128.

## Next Integration Point

The next code change belongs in upstream Megatron's
`TensorParallelMuon.step` / `TensorParallelMuon.orthogonalize` path:

```text
/home/dave/megatron-lm/megatron/core/optimizer/emerging_optimizers.py
```

Exact staged plan:

1. Add an explicit `muon_ns_carrier` optimizer option on `TensorParallelMuon`
   with values like `bf16`, `mxfp8_probe`, and later `mxfp8`. Keep the default
   at `bf16`.
2. In `TensorParallelMuon.step`, branch at the current grouped qMuon call:

   ```text
   quantized_muon_momentum_update_multi_and_normalize_groups_(...)
   ```

   For carrier mode, replace it with a grouped carrier emitter that returns a
   `QuantizedMuonMxfp8Carrier` per logical 2D tensor or QKV group. This emitter
   must update the q8 state and produce carrier payload/scales without writing
   normalized BF16 values into `batch["grads"]`.
3. Thread the carrier into orthogonalization explicitly instead of pretending it
   is a grad tensor. A practical interface is either
   `orthogonalize_carrier(p, carrier, ...)` or an optional `carrier=` argument
   on `orthogonalize`.
4. In `TensorParallelMuon.orthogonalize`, handle QKV split before calling NS:
   carrier row slices must follow the same logical Q/K/V groups as
   `_quantized_momentum_norm_segments`, with block-boundary validation matching
   `build_quantized_muon_norm_plan`.
5. In `scaled_orthogonalize_fn`, route carrier mode to a new
   `_newton_schulz_mxfp8_carrier` implementation. First milestone: reproduce
   the probe's carrier-fed Gram path in the optimizer. Second milestone:
   remove the current BF16 GEMM output by adding an FP16/FP32 output variant or
   immediate MXFP8 requantized output for NS intermediates.
6. Fail closed. If a shape, segment boundary, scale layout, or TP shard cannot
   stay on the carrier path, raise or increment an explicit fallback audit
   counter. Do not silently return to the BF16 owned path.

## Acceptance Gate

Full model acceptance still requires a locked real-data run that actually uses
the carrier in optimizer training:

```text
flock /tmp/cppmega_gpu_profile.lock <6-step training command> --muon-dtype-audit --muon-ns-carrier mxfp8
```

Required counters:

```text
bf16_owned_path_observed=0
qmuon_grad_dtype_bfloat16_tensors=0
ns_step_dtype_bfloat16_tensors=0
ns_lowmem_output_dtype_bfloat16_tensors=0
qmuon_mxfp8_carrier_update_calls>0
```

Required behavior:

- finite loss and finite grads through the 6-step run;
- no silent BF16 fallback;
- faster than the Wave15A BF16 baseline (`5075.5 ms/iter`) or a clear measured
  explanation of where the integration lost the 1.842x microprobe win.

Current recommendation: merge only as a staged probe/audit path. Hold or reject
claims of full optimizer acceptance until the `TensorParallelMuon` carrier route
passes the gate above.
