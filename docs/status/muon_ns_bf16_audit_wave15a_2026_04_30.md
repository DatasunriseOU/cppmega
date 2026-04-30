# Wave15A Muon/Newton-Schulz BF16 Audit - 2026-04-30

Goal: remove BF16 from the owned Muon/q8 momentum/Newton-Schulz optimizer path,
or prove where it remains. The hard gate was no BF16 fallback/materialization in
the owned path and faster hot steps than the BF16 baseline.

## Source Findings

- `cppmega/megatron/quantized_muon_momentum.py` documents the boundary as BF16:
  callers feed the returned BF16 scratch into low-memory Newton-Schulz.
- `quantized_muon_momentum_update_` allocates BF16 scratch when the caller does
  not pass one, and `_validate_update_inputs` requires scratch dtype BF16.
- The production q8 multi-update path overwrites `grad` in place and explicitly
  treats BF16 grads as the low-memory Newton-Schulz input.
- `cppmega/megatron/cuda_ext/quantized_muon_momentum.cu` specializes the update
  kernel for BF16 grads and stores updated momentum back into the grad buffer
  with `store_value<__nv_bfloat16>`.
- Megatron's `TensorParallelMuon.step` q8 branch calls
  `quantized_muon_momentum_update_multi_and_normalize_groups_` and then
  `orthogonalize(..., already_normalized=True)`. In that branch,
  `scaled_orthogonalize_fn` selects `_newton_schulz_tp_lowmem` directly, so the
  typed `muon_fp32_matmul_prec=high` knob does not move q8 NS off the BF16 grad.
- The current Megatron CLI contract also pins this lane to BF16 main grads:
  `--main-grads-dtype` only accepts `bf16`, and `--muon-quantized-momentum`
  requires BF16 training plus the BF16 no-master emerging optimizer.
- `nanochat/muon.py` is BF16-oriented for Polar Express/Newton-Schulz and did
  not provide a qMuon drop-in.
- TransformerEngine has MXFP8 cast/GEMM helpers, but no existing fused
  dequant->Newton-Schulz update helper that avoids materializing the NS input
  tensor between iterations.

## Added Audit

`cppmega.megatron.muon_dtype_audit` installs observational wrappers around the
cppmega qMuon update helpers and Megatron's low-memory NS helpers. It prints one
greppable line at exit, for example:

```text
[cppmega_muon_dtype_audit] bf16_owned_path_observed=1 ...
```

The local GB10 launcher exposes it through typed profile/CLI parameters:

- `--muon-dtype-audit`
- `--muon-fp32-matmul-prec {low,medium,high}`

## Locked GPU Runs

All GPU runs below used `flock /tmp/cppmega_gpu_profile.lock`.

| Run | Log | Hot steps 3-6 | Tok/s | Peak torch max alloc | BF16 owned path |
| --- | --- | ---: | ---: | ---: | --- |
| BF16/tensorwise medium NS | `/home/dave/logs/wave15A_bf16_medium_20260430_001.log` | 4871.175 ms | 3363.5 | 27.500 GiB | yes |
| BF16/tensorwise `muon-fp32-matmul-prec=high` | `/home/dave/logs/wave15A_bf16_high_20260430_001.log` | 4894.300 ms | 3347.6 | 27.498 GiB | yes |
| MXFP8 TE-TN medium NS | `/home/dave/logs/wave15A_mxfp8_tetn_20260430_001.log` | 5287.550 ms | 3098.6 | 26.154 GiB | yes |

The BF16-high probe was not faster than BF16 medium, and the audit counters were
identical for BF16 qMuon grads and BF16 NS steps. MXFP8 TE-TN had
`bf16_fallback_dgrad=0` and `bf16_fallback_wgrad=0`, but still had
`qmuon_grad_dtype_bfloat16_tensors=2580`,
`ns_step_dtype_bfloat16_tensors=4086`, and `bf16_owned_path_observed=1`.

## Recommendation

Reject for merge as a candidate optimizer replacement. This work proves the
remaining BF16 is not a TE fallback; it is the owned qMuon/NS contract. Removing
it needs a new or upstreamed kernel path that keeps the updated q8 momentum and
Newton-Schulz iteration state out of BF16, or a different optimizer contract for
main grads. Existing MXFP8 TE storage and the current typed matmul-precision knob
do not satisfy the gate.
