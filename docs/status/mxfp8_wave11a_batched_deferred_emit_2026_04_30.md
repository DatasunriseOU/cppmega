# MXFP8 Wave11A Batched Deferred Emit Status - 2026-04-30

## Target

Wave11A tested whether the TE Linear autograd save path can reduce MXFP8
per-step launch/materialization overhead by batching deferred rowwise-transpose
emits and copy-transposes. The candidate does not add producer sidecars. It
queues TE saved-transpose operands, flushes them before MXFP8 backward GEMMs,
and keeps the profile control as typed config:

```bash
--mxfp8-deferred-emit-batching
--mxfp8-deferred-emit-max-pending-mib
--mxfp8-deferred-emit-max-pending-operands
```

The feature remains default-off because the accepted criterion is MXFP8 faster
than BF16, not just fewer materialization launches.

## Implementation

- `cppmega/megatron/mxfp8_batched_transpose.py` adds a local CUDA extension
  for batched BF16-to-MXFP8 rowwise transpose emit and uint8 payload
  transpose.
- `scripts/cppmega_fp8_shim.py` patches TE deferred save helpers only when
  batching is enabled, queues operands, shape-groups flushes, and flushes
  before dense/grouped MXFP8 backward GEMMs.
- `cppmega/recipes/run_profiles.py` adds typed `PrecisionProfile` fields and
  CLI flags for the opt-in batching path.
- `scripts/local_gb10_quarter_train.sh` exports default-off launcher knobs.
- `tools/probes/gb10_accepted_path_validation_helpers.py` parses the new
  batching counters from training logs.

## Tests

CPU-only checks:

```text
/home/dave/cppmega-venv/bin/python -m py_compile cppmega/recipes/run_profiles.py scripts/cppmega_fp8_shim.py cppmega/megatron/mxfp8_batched_transpose.py tests/test_mxfp8_batched_transpose.py tools/probes/gb10_accepted_path_validation_helpers.py
/home/dave/cppmega-venv/bin/python -m pytest tests/test_run_profiles.py -q
/home/dave/cppmega-venv/bin/python -m pytest tests/test_gb10_accepted_path_validation.py -q
/home/dave/cppmega-venv/bin/python -m pytest tests/test_mxfp8_sidecar_lifecycle.py tests/test_cutlass_mxfp8_shim_routing.py tests/test_grouped_mxfp8_direct_routing.py -q
```

Locked GPU checks used `flock /tmp/cppmega_gpu_profile.lock`:

```text
flock /tmp/cppmega_gpu_profile.lock /home/dave/cppmega-venv/bin/python -m pytest tests/test_mxfp8_batched_transpose.py -q
```

## Real-Data Six-Step Runs

All 6-step GPU/profile runs used `flock /tmp/cppmega_gpu_profile.lock`.

| Run | Log | Hot avg steps 3-6 | Tok/s | Max alloc |
| --- | --- | ---: | ---: | ---: |
| Fresh BF16 | `/home/dave/logs/wave11a_bf16_6step_20260430_010225.log` | 5071.475 ms | 3230.628 | 27.229 GiB |
| Current MXFP8 | `/home/dave/logs/wave11a_mxfp8_current_6step_20260430_010954.log` | 5310.800 ms | 3085.080 | 26.317 GiB |
| Naive batched MXFP8, 16 operands | `/home/dave/logs/wave11a_mxfp8_batched_6step_20260430_011329.log` | 5378.950 ms | 3045.970 | 26.336 GiB |
| Shape-grouped batched MXFP8, 64 operands | `/home/dave/logs/wave11a_mxfp8_batched_grouped_commit_20260430_013334.log` | 5269.450 ms | 3109.268 | 26.332 GiB |

Shape-grouped batching improved current MXFP8 by 41.350 ms per hot step
(about 0.78%), but it remained 197.975 ms slower than fresh BF16 (3.90%).
Naive arbitrary batching was slower than both current MXFP8 and BF16.

## Counters

Current MXFP8:

```text
mxfp8_tn_adapter_dgrad=264
mxfp8_tn_adapter_wgrad=264
mxfp8_tn_adapter_saved_transpose_operand=2808
mxfp8_tn_adapter_copy_transpose=684
mxfp8_tn_adapter_missing_sidecar_copy=684
mxfp8_grouped_transpose_copy_fallback_dgrad=60
mxfp8_grouped_transpose_copy_fallback_wgrad=60
mxfp8_batched_transpose_flushes=0
mxfp8_batched_transpose_operands=0
```

Naive batched MXFP8:

```text
mxfp8_batched_transpose_flushes=270
mxfp8_batched_transpose_operands=2808
mxfp8_batched_transpose_bf16_emit_operands=1578
mxfp8_batched_transpose_uint8_copy_operands=1230
mxfp8_batched_transpose_max_pending=16
mxfp8_batched_transpose_pending_bytes_peak=503635968
mxfp8_batched_transpose_flush_failures=0
```

Shape-grouped batched MXFP8:

```text
mxfp8_batched_transpose_flushes=1748
mxfp8_batched_transpose_operands=2808
mxfp8_batched_transpose_bf16_emit_operands=1578
mxfp8_batched_transpose_uint8_copy_operands=1230
mxfp8_batched_transpose_max_pending=64
mxfp8_batched_transpose_pending_bytes_peak=1088575488
mxfp8_batched_transpose_flush_failures=0
```

The grouped variant reduced overhead enough to beat the current MXFP8 baseline
but did not beat BF16. The higher flush count is intentional: grouping avoids
mixed-shape launch waste from the naive batched kernel.

## Recommendation

Reject Wave11A for acceptance/default. Keep the implementation as an opt-in
profiling path with `mxfp8_deferred_emit_batching=False` by default. It is a
useful measurement harness for deferred emit batching, but it does not satisfy
the goal of MXFP8 faster than BF16.

The next viable target is still a producer/autograd contract that saves
GEMM-ready MXFP8 operands directly, eliminating the per-call TE emit/copy work
instead of moving it into batched flushes.
