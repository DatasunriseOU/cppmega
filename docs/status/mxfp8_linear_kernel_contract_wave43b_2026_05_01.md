# MXFP8 Linear Kernel Contract Wave43B - 2026-05-01

Status: active
Canonical: docs/status/cppmega_run_profiles_and_token_flow.md
Date: 2026-05-01
Scope: strict direct compact-columnwise Linear backward routing guardrails.

## What Changed

Wave43B adds a typed run-profile knob:

```bash
--mxfp8-linear-kernel-contract legacy|gemm_ready_v1|gemm_ready_v1_dense_only
```

The rendered `CPPMEGA_TE_MXFP8_LINEAR_KERNEL_CONTRACT` remains only the
import-time transport into `scripts/cppmega_fp8_shim.py`. Launchers and tests
must set the dataclass/CLI field, not a raw env-only switch.

## Contract Semantics

- `legacy`: current measured default. TE-TN/FlashInfer/CUTLASS routes may use
  saved rowwise-transpose operands, sidecars, or copy-transpose fallbacks.
- `gemm_ready_v1`: strict dense plus grouped contract. Dense Linear backward
  must consume compact-columnwise/direct operands, and grouped backward must
  consume GEMM-ready grouped operands. Any copy-transpose fallback raises.
- `gemm_ready_v1_dense_only`: strict dense Linear contract only. Grouped/MoE
  fallback remains available but is counted as excluded from v1 coverage.

Selecting a v1 contract forces dense saved operands on, compact-columnwise
backward on, and transpose emit off inside the shim. That makes missed direct
support fail at the first Linear backward call instead of silently measuring a
materialized transpose path.

## Acceptance Counters

A successful strict dense run should show positive direct counters and zero
bridge counters:

```text
mxfp8_linear_contract_v1_dense_dgrad>0
mxfp8_linear_contract_v1_dense_wgrad>0
mxfp8_linear_contract_v1_dense_miss_dgrad=0
mxfp8_linear_contract_v1_dense_miss_wgrad=0
mxfp8_tn_adapter_copy_transpose=0
mxfp8_tn_adapter_saved_transpose_operand=0
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
```

For full `gemm_ready_v1`, grouped/MoE must also avoid the grouped transpose
fallback:

```text
mxfp8_linear_contract_v1_grouped_miss_dgrad=0
mxfp8_linear_contract_v1_grouped_miss_wgrad=0
mxfp8_grouped_transpose_copy_fallback_dgrad=0
mxfp8_grouped_transpose_copy_fallback_wgrad=0
```

## Donor/Upstream Layout Notes

The useful upstream direction is still producer-side GEMM-ready layout, not
another Python wrapper. FlashInfer/CUTLASS SM120 MXFP8 paths are built around a
native/swizzled scale layout such as `SWIZZLED_128x4`, while TE's MXFP8 tensors
expose compact rowwise/columnwise payloads for its own GEMM descriptors. The
contract added here does not solve that layout mismatch; it makes the mismatch
visible by rejecting the old deferred copy/transpose bridge.

## Validation

Non-GPU validation in this worktree:

```text
/home/dave/cppmega-venv/bin/python -m py_compile \
  cppmega/recipes/run_profiles.py \
  scripts/cppmega_fp8_shim.py \
  tests/test_mxfp8_linear_kernel_contract.py \
  tests/test_grouped_mxfp8_direct_routing.py \
  tests/test_run_profiles.py

/home/dave/cppmega-venv/bin/python -m pytest --confcutdir=tests \
  tests/test_run_profiles.py \
  tests/test_grouped_mxfp8_direct_routing.py \
  tests/test_mxfp8_linear_kernel_contract.py -q

27 passed, 7 skipped
```

The skipped shim tests require the host MXFP8/TE import path. No shared
TransformerEngine checkout or shared venv/install was modified by Wave43B.

Locked GPU smoke attempted on this host:

```text
flock /tmp/cppmega_gpu_profile.lock timeout 1200 \
  scripts/local_gb10_quarter_train.sh \
  --train-iters 20 \
  --fp8-recipe mxfp8 \
  --mxfp8-bwd-backend flashinfer_cutlass \
  --mxfp8-linear-kernel-contract gemm_ready_v1_dense_only \
  --mem-profile \
  --mem-profile-steps 2
```

It did not reach model construction: the typed profile prepended
`/home/dave/TransformerEngine`, and that checkout currently fails import with
`FileNotFoundError: Could not find shared object file for Transformer Engine core lib.`
Wave43B therefore adds typed runtime source-root overrides so a later run can
select a valid installed package or source checkout without raw environment
switches:

```bash
--transformer-engine-source ''
--flash-attention-source /path/to/flash-attn
```

Repeating the smoke with `--transformer-engine-source '' --flash-attention-source ''`
still imported `/home/dave/TransformerEngine`, which means the shared venv itself
currently resolves TE to that broken editable/source checkout. Per Wave43B scope,
the worktree did not rebuild TE, uninstall packages, or mutate the shared venv.

No tok/s, Nsight, torch-profiler, or memory-profiler metrics were produced from
that blocked run.
