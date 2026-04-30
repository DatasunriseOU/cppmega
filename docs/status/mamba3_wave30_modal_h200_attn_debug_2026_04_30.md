# Mamba3 Wave30 Lane A Modal H200 Attention Debug - 2026-04-30

Branch: `worker/mamba3-wave30-modal-h200-attn-debug`

Base: `cceeeeb`

Scope: Modal-only H200:2 attention-debug and Mamba backward reachability harness.
No production model/kernel default was changed.

## Harness

`scripts/modal_mamba3_wave30_h200_attn_debug.py`

- Uses GHCR image `ghcr.io/jewelmusicee/cppmega:785c3fd`.
- Uses Modal GPU spec `H200:2`.
- Writes artifacts under `/vol/benchmarks/mamba3_wave30_modal_h200_attn_debug`.
- Enables `NVTE_DEBUG=1` and `NVTE_DEBUG_LEVEL=2` in `debug_sweep`.
- Captures TE attention backend lines, Mamba backward markers, peak memory, log tails,
  command JSON, and result summaries.
- Keeps fallback cases labeled as debug/profiling reachability only, not
  production-throughput claims.
- Rolls back the stage2 file mutation in `finally`; final kernel state was clean.

## Commands

```bash
python -m py_compile scripts/modal_mamba3_wave30_h200_attn_debug.py

modal app list --json

modal run scripts/modal_mamba3_wave30_h200_attn_debug.py::launch_debug_sweep \
  --run-id wave30_h200_attn_debug_20260430 \
  --train-iters 1 \
  --timeout-per-case-s 1800

modal app stop ap-PhQfXe7rrGY6AWrysqz0QN

modal run scripts/modal_mamba3_wave30_h200_attn_debug.py::launch_debug_sweep \
  --run-id wave30_h200_auto_stage2_reach_20260430 \
  --train-iters 1 \
  --cases fallback_auto_full \
  --timeout-per-case-s 1800

modal app stop ap-8VcHJjE3E7lAZZCR4Mk6fH

modal run scripts/modal_mamba3_wave30_h200_attn_debug.py::launch_debug_sweep \
  --run-id wave30_h200_auto_stage2_reach2_20260430 \
  --train-iters 1 \
  --cases fallback_auto_full \
  --timeout-per-case-s 1800

modal app stop ap-hxoZoaHXfXO1BlJyWE2Ivn

modal volume get cppmega-mamba3-benchmarks \
  /benchmarks/mamba3_wave30_modal_h200_attn_debug/wave30_h200_attn_debug_20260430 \
  artifacts/mamba3_wave30_modal_h200_attn_debug --force

modal volume get cppmega-mamba3-benchmarks \
  /benchmarks/mamba3_wave30_modal_h200_attn_debug/wave30_h200_auto_stage2_reach2_20260430 \
  artifacts/mamba3_wave30_modal_h200_attn_debug --force
```

## Modal Apps

| app id | description | status |
| --- | --- | --- |
| `ap-PhQfXe7rrGY6AWrysqz0QN` | `cppmega-wave30-modal-h200-attn-debug` fallback sweep | stopped; explicit stop returned already stopped |
| `ap-8VcHJjE3E7lAZZCR4Mk6fH` | focused `fallback_auto_full` rerun | stopped; explicit stop returned already stopped |
| `ap-hxoZoaHXfXO1BlJyWE2Ivn` | focused `fallback_auto_full` baseline vs stage2 rerun | stopped; explicit stop returned already stopped |

`modal app list --json` after the runs showed no live Wave30 app.

## Exact TE Attention Reason

Production-shape `te_flash_full` reproduced the Wave29 blocker before
iteration 1. With `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2`, TE reported:

```text
DEBUG:DotProductAttention:Disabling FusedAttention due to NVTE_FUSED_ATTN=0
DEBUG:DotProductAttention:Disabling UnfusedDotProductAttention due to NVTE_UNFUSED_ATTN=0
DEBUG:DotProductAttention:Disabling FlashAttention 2 as it does not support MLA.
DEBUG:DotProductAttention:Available backends = {FlashAttention=False, FusedAttention=False, UnfusedDotProductAttention=False}
```

So the blocker is not H200 availability. `--attention-backend flash` forces the
non-flash TE backends off, while FlashAttention 2 rejects MLA.

## Results

Dataset: `synthetic_full_shape_mock_data`; real production indexed data was not
present on the Modal volume.

Artifacts:

- `artifacts/mamba3_wave30_modal_h200_attn_debug/wave30_h200_attn_debug_20260430/`
- `artifacts/mamba3_wave30_modal_h200_attn_debug/wave30_h200_auto_stage2_reach2_20260430/`

| case | variant | production throughput | seq | attention path | reached Mamba backward | peak alloc GiB | status |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| `te_flash_full` | baseline | yes | 4096 | flash forced; all TE backends unavailable | no | 16.983 | attention blocker |
| `fallback_auto_full` | baseline | no | 4096 | auto selected FusedAttention sub-backend 1 | yes | 37.946 | `G value of 8 is not currently supported!` |
| `fallback_unfused_no_flash_full` | baseline | no | 4096 | UnfusedDotProductAttention | yes | 48.881 | `G value of 8 is not currently supported!` |
| `fallback_auto_no_flash_seq2048` | baseline | no | 2048 | auto selected FusedAttention sub-backend 1 | yes | 29.642 | `G value of 8 is not currently supported!` |
| `fallback_auto_full` | `stage2_force_nontma_bf1_bb0` | no | 4096 | auto selected FusedAttention sub-backend 1 | yes | 37.946 | `G value of 8 is not currently supported!` |

No case completed an iteration, so there is no valid tok/sec. The useful timing
available is wall-clock reachability timing in `result.json`: first full auto
baseline 272.908 s; final focused auto baseline 279.355 s; focused auto stage2
189.511 s. These include startup/JIT and are not throughput metrics.

Stage2 did apply in the final focused run:

```text
disable_tma_count=13
flat_q=true
flat_qk=true
bf_num_stages_1=true
bb_num_stages_0=true
```

The final rollback restored a clean baseline kernel:

```text
disable_tma_count=0
flat_q=false
flat_qk=false
bf_num_stages_1=false
bb_num_stages_0=true
```

## Next Patch

Patch `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py` and the cppmega
stage2 applier to support `G=8` in `mamba_mimo_bwd_combined`. The attention
route is now clear with `attention_backend=auto`; both baseline and stage2 reach
Mamba backward, then stop at the same hard-coded group-count guard. After that
guard is removed correctly, rerun `fallback_auto_full` for 1 step, then run the
20-step `fallback_auto_full` baseline vs stage2 gate. Keep it labeled as a
fallback until the production recipe changes from forced flash to a supported
MLA backend policy.
