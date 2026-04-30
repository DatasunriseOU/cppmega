# Mamba3 Wave29 Lane C H100 - 2026-04-30

Branch: `worker/mamba3-wave29-tilelang-prod-tune`

Base: Wave28 worker branch after the guarded stage2 force-nonTMA production
control and partial-marker fix.

Scope: safe production-path work only. No production patch semantics changed;
the current candidate remains `bf_num_stages=1, bb_num_stages=0`.

## Changes

- Hardened the stage2 applier local-rank wait path. Non-rank-0 workers now
  require both the local sentinel and the expected installed-source state
  before proceeding, so a stale lockfile from a previous apply/rollback run
  cannot be accepted blindly.
- Added unit coverage for the applied/absent state predicates and partial-state
  rejection.
- Added `scripts/modal_mamba3_wave29_lane_c_h100.py`, a production-focused H100
  harness that compares only baseline vs current guarded stage2, records
  allocated/reserved/free memory deltas, and supports optional split-kernel
  NVTX ranges via `--profile-nvtx`.

## Commands

```text
python -m py_compile \
  cppmega/megatron/upstream_patches/apply_mamba3_stage2_force_nontma_patches.py \
  scripts/modal_mamba3_wave29_lane_c_h100.py \
  tests/test_mamba3_stage2_force_nontma_applier.py

pytest -q tests/test_mamba3_stage2_force_nontma_applier.py

PYTHONPATH=. python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches

PYTHONPATH=. CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1 \
  python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches

modal run scripts/modal_mamba3_wave29_lane_c_h100.py --verify-applier

modal run scripts/modal_mamba3_wave29_lane_c_h100.py \
  --run-id wave29_lane_c_h100_20260430_3

modal run scripts/modal_mamba3_wave29_lane_c_h100.py \
  --run-id wave29_lane_c_h100_20260430_4
```

Modal app name: `cppmega-wave29-lane-c-h100`

Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

GPU: `NVIDIA H100 80GB HBM3`, capability `(9, 0)`.

Artifacts:

- `/benchmarks/mamba3_wave29_lane_c_h100/wave29_lane_c_h100_20260430_3/report.json`
- `/benchmarks/mamba3_wave29_lane_c_h100/wave29_lane_c_h100_20260430_4/report.json`

## Applier Verification

Default-off:

```text
SKIP CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA is not set
```

Mutation gate:

```text
FAIL: Refusing to mutate installed mamba_ssm without MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1
```

H100 source mutation check:

```json
{
  "apply_returncode": 0,
  "rollback_returncode": 0,
  "restored_original_bytes": true,
  "patched": {
    "flat_q": true,
    "flat_qk": true,
    "bf_num_stages_1": true,
    "bb_num_stages_0": true,
    "disable_tma_count": 13
  }
}
```

## Results

Run `wave29_lane_c_h100_20260430_3`:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | peak alloc MiB delta | peak reserved MiB delta | max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke B1 S128 H4 | baseline | 0.0388 | 0.0822 | 0.6648 | 1.21 | 0.00 | 0.0 |
| smoke B1 S128 H4 | stage2 current | 0.0427 | 0.0824 | 0.6435 | 1.21 | 0.00 | 0.0 |
| rep B2 S512 H8 | baseline | 0.1352 | 0.3160 | 1.0420 | 33.04 | 38.00 | 0.0 |
| rep B2 S512 H8 | stage2 current | 0.1380 | 0.3150 | 0.8728 | 33.04 | 42.00 | 0.0 |

Run `wave29_lane_c_h100_20260430_4`:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | peak alloc MiB delta | peak reserved MiB delta | max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke B1 S128 H4 | baseline | 0.0363 | 0.0804 | 0.3974 | 1.21 | 0.00 | 0.0 |
| smoke B1 S128 H4 | stage2 current | 0.0371 | 0.0809 | 0.3547 | 1.21 | 0.00 | 0.0 |
| rep B2 S512 H8 | baseline | 0.1311 | 0.3122 | 0.6315 | 33.04 | 38.00 | 0.0 |
| rep B2 S512 H8 | stage2 current | 0.1356 | 0.3119 | 0.9333 | 33.04 | 42.00 | 0.0 |

Correctness:

- `stage2_current_vs_baseline`: every non-`None` output had `max_abs=0.0` in
  both runs.

Memory:

- Peak allocated memory was unchanged by the current stage2 candidate on both
  shapes.
- Representative peak reserved memory was 4 MiB higher for stage2 current in
  both repeats. This is allocator reservation, not live allocated tensor memory.

## Judgment

No new kernel production patch from Wave29. The current guarded `bf=1,bb=0`
candidate remains correct on H100 component shapes, but H100 split-kernel timing
does not show a new production win: `bwd_bwd` is flat, `bwd_fwd` is slightly
slower in these repeats, and chain timing is too noisy to attribute.

Safe for main:

- Yes: applier stale-sentinel hardening, tests, and Wave29 H100/NVTX harness.
- No: any change to the stage2 TileLang patch semantics based on this wave.
