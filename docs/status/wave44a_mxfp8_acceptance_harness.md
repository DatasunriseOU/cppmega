# Wave44A MXFP8 Acceptance Harness

This worktree adds a typed command/report harness for local GB10 BF16 vs MXFP8
20-step acceptance runs. The harness is intentionally profile/dataclass driven:
it builds a `RunProfile`, mutates typed fields, renders the existing profile
shell assignments, and then wraps the launch with
`flock /tmp/cppmega_gpu_profile.lock`.

## Commands

Plan all runs without touching the GPU:

```bash
/home/dave/cppmega-venv/bin/python tools/profiling/mxfp8_acceptance_harness.py \
  plan --steps 20 --run-id-prefix wave44a --format shell
```

Run only the two 20-step train/memory lanes:

```bash
/home/dave/cppmega-venv/bin/python tools/profiling/mxfp8_acceptance_harness.py \
  run --steps 20 --run-id-prefix wave44a --only train
```

Run profiler lanes separately, because CUPTI rejects overlapping torch profiler
and Nsight subscribers:

```bash
/home/dave/cppmega-venv/bin/python tools/profiling/mxfp8_acceptance_harness.py \
  run --steps 20 --run-id-prefix wave44a --only profilers
```

Compare completed train logs:

```bash
/home/dave/cppmega-venv/bin/python tools/profiling/mxfp8_acceptance_harness.py \
  compare \
  --bf16-log /home/dave/logs/wave44a_bf16_train_20step_<stamp>.log \
  --mxfp8-log /home/dave/logs/wave44a_mxfp8_train_20step_<stamp>.log \
  --hot-step-start 3 --format table
```

## Metrics

The report includes:

- steady/hot-step `ms/iter`
- `tok/sec`
- max allocated and max reserved GiB/bytes
- train, validation, and test loss
- skipped and NaN iteration counts
- parameter storage breakdown
- MXFP8 acceptance counters, including BF16 fallback, copy-transpose, and
  sidecar registry counters
- referenced torch/nsys/ncu artifacts

## Acceptance Counters

The MXFP8 lane is not accepted unless these stay zero:

- `bf16_fallback_dgrad`
- `bf16_fallback_wgrad`
- `mxfp8_tn_adapter_copy_transpose`
- `mxfp8_tn_sidecar_registry_peak`
- `mxfp8_tn_adapter_saved_transpose_operand`

Direct or GEMM-ready producer counters should be non-zero. Missing producer API
counters should stay zero.

## Validation

```bash
/home/dave/cppmega-venv/bin/python -m py_compile \
  tools/profiling/mxfp8_acceptance_harness.py \
  tools/profiling/compare_bf16_mxfp8.py
/home/dave/cppmega-venv/bin/python -m pytest --confcutdir=tests \
  tests/test_mxfp8_acceptance_harness.py \
  tests/test_compare_bf16_mxfp8.py \
  tests/test_profile_report.py -q
```

Current validation in this worktree: `6 passed`.
