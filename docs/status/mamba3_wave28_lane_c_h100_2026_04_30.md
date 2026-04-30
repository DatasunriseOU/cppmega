# Mamba3 Wave28 Lane C H100 - 2026-04-30

Branch: `worker/mamba3-wave28-tilelang-prod-tune`

Base: `d7513ec` (`perf(mamba3): add guarded stage2 force-nontma control`)

Scope: production TileLang stage2/backward path around the guarded
`bf_num_stages=1, bb_num_stages=0` candidate. H100 only; no H200 mini runs.

## Commands

```text
python -m py_compile \
  cppmega/megatron/upstream_patches/apply_mamba3_stage2_force_nontma_patches.py \
  scripts/modal_mamba3_wave28_lane_c_h100.py \
  tests/test_mamba3_stage2_force_nontma_applier.py

pytest -q tests/test_mamba3_stage2_force_nontma_applier.py

modal run scripts/modal_mamba3_wave28_lane_c_h100.py \
  --run-id wave28_lane_c_h100_20260430_3

modal run scripts/modal_mamba3_wave28_lane_c_h100.py \
  --run-id wave28_lane_c_h100_20260430_4

modal run scripts/modal_mamba3_wave28_lane_c_h100.py --verify-applier

modal app list
```

Modal app name: `cppmega-wave28-lane-c-h100`

Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

GPU: `NVIDIA H100 80GB HBM3`, capability `(9, 0)`

Artifacts:

- `/benchmarks/mamba3_wave28_lane_c_h100/wave28_lane_c_h100_20260430_3/report.json`
- `/benchmarks/mamba3_wave28_lane_c_h100/wave28_lane_c_h100_20260430_4/report.json`

All Lane C Modal apps observed by `modal app list` were `stopped`, `Tasks=0`.

## Applier Verification

Default-off local check:

```text
PYTHONPATH=. python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
# SKIP CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA is not set
```

Mutation-gate local check:

```text
PYTHONPATH=. CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1 \
  python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
# FAIL: Refusing to mutate installed mamba_ssm without MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1
```

H100 container source-present check after fixing the partial-marker guard:

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

Bug fixed: the old partial-marker guard treated clean installed sources as
partial because upstream already has `bb_num_stages=0`. The guard now keys
partial detection off structural stage2 markers (`flat_q`, `flat_qk`,
`direct_qk`) before requiring all patch markers.

## Candidate Tested

`lane_c_shared_reuse`: keep the bwd_bwd shared `[chunk_size]` vectors because
they are reused later, but remove two first-use fragment stages:

- `dA_cs_rev_frag` first consumer reads `dA_cs_rev_shared[csr//R]`
- `dA_cs_dq_frag` first consumer reads `dA_cs_shared[csr//R]`

An earlier direct-fragment variant was rejected because deleting the shared
vectors broke later consumers (`NameError: dA_cs_rev_shared is not defined`).

## Results

Run `wave28_lane_c_h100_20260430_3`:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | peak MiB delta | max diff vs current |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke B1 S128 H4 | baseline | 0.0368 | 0.0811 | 0.4830 | 1.21 | 0.0 |
| smoke B1 S128 H4 | stage2 current | 0.0378 | 0.0812 | 0.4844 | 1.21 | 0.0 |
| smoke B1 S128 H4 | lane_c_shared_reuse | 0.0390 | 0.0811 | 0.4789 | 1.21 | 0.0 |
| rep B2 S512 H8 | baseline | 0.1330 | 0.3156 | 0.6380 | 33.04 | 0.0 |
| rep B2 S512 H8 | stage2 current | 0.1370 | 0.3145 | 0.7072 | 33.04 | 0.0 |
| rep B2 S512 H8 | lane_c_shared_reuse | 0.1381 | 0.3147 | 0.6809 | 33.04 | 0.0 |

Run `wave28_lane_c_h100_20260430_4`:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | peak MiB delta | max diff vs current |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke B1 S128 H4 | baseline | 0.0370 | 0.0813 | 0.3785 | 1.21 | 0.0 |
| smoke B1 S128 H4 | stage2 current | 0.0375 | 0.0817 | 0.3566 | 1.21 | 0.0 |
| smoke B1 S128 H4 | lane_c_shared_reuse | 0.0373 | 0.0818 | 0.3201 | 1.21 | 0.0 |
| rep B2 S512 H8 | baseline | 0.1320 | 0.3126 | 0.6205 | 33.04 | 0.0 |
| rep B2 S512 H8 | stage2 current | 0.1359 | 0.3116 | 0.6976 | 33.04 | 0.0 |
| rep B2 S512 H8 | lane_c_shared_reuse | 0.1362 | 0.3119 | 0.6543 | 33.04 | 0.0 |

Correctness:

- `stage2_current_vs_baseline`: every non-`None` output had `max_abs=0.0`.
- `lane_c_shared_reuse_vs_stage2_current`: every non-`None` output had `max_abs=0.0`.

## Judgment

No production kernel patch from Lane C. The candidate is correct, but the
focused split-kernel evidence does not show a real bwd_bwd win, and peak memory
is unchanged. The apparent chain improvement is not attributable to the edited
bwd_bwd path because split bwd_bwd is flat/slightly slower across repeat runs.

Safe for main:

- Yes: applier partial-marker guard fix and H100 harness/docs.
- No: changing the stage2 TileLang kernel patch to `lane_c_shared_reuse`.

Keep the current guarded production candidate as `bf=1,bb=0`; continue avoiding
bwd_bwd WS/TMA unless a future counter-level profile shows a clear reason.
