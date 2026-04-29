# Mamba3 Stage2 Force Non-TMA Smem-Safe Fix - 2026-04-29

Branch: `worker/mamba3-stage2-force-nontma`

Goal: fix the H200 productionish crash from the stage2 force-nonTMA benchmark.

## Problem

The previous `stage2_force_nontma` benchmark used `bf_num_stages=2` and
`bb_num_stages=2`. It compiled and enabled WS, but crashed on the productionish
shape at launch time:

```text
InternalError: Failed to set the allowed dynamic shared memory size to 231712
```

That is a Hopper dynamic shared memory limit issue, not a correctness failure.

## Fix

Set the stage2 force-nonTMA path to `num_stages=1` for both backward kernels:

- patch default `bf_num_stages`: `2 -> 1`
- patch default `bb_num_stages`: `2 -> 1`
- benchmark variant `bf_num_stages`: `2 -> 1`
- benchmark variant `bb_num_stages`: `2 -> 1`

The earlier smoke matrix already showed that `(1,1)` still triggers WS in both
`bwd_fwd` and `bwd_bwd`, so this keeps the intended TMA/WS path while reducing
shared memory enough for the productionish launch.

## Validation

Local:

```text
python -m py_compile scripts/modal_mamba3_stage2_force_nontma_benchmark.py scripts/modal_mamba3_stage2_force_nontma_probe.py
patch --dry-run -p4 /tmp/.../mamba3_mimo_bwd.py < upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_force_nontma.patch
git diff --check
```

H200 Modal run:

```text
GHCR_TAG=785c3fd \
CPPMEGA_MODAL_GPU=H200:2 \
timeout 900s \
modal run scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_force_nontma_h200_smemsafe_20260429_1 \
  --shape-csv productionish \
  --warmup 1 \
  --iters 4
```

App:

- `ap-oN7Dd2Gr0tyXeZubgqAVbn`
- state after run: stopped, `Tasks=0`

Artifacts in Modal Volume `cppmega-mamba3-benchmarks`:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_smemsafe_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_smemsafe_20260429_1/summary.csv`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_smemsafe_20260429_1/summary.json`

Device:

- GPU: `NVIDIA H200`
- device count: `2`
- capability: `(9, 0)`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

## Productionish Result

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | WS | TMA loads | status |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| baseline | 1.8847 | 3.7181 | 5.5636 | no/no | 0/0 | ok |
| stage2_force_nontma `(1,1)` | 1.8043 | 3.9903 | 5.7865 | yes/yes | 5/7 | ok |

Correctness against baseline:

- `max_main_grad_abs_diff`: `1.1742660177560538e-09`
- `qk_dot` max diff: `0.0`
- `states` max diff: `0.0`

Speed ratios versus baseline:

- `bwd_fwd`: `1.0446x`
- `bwd_bwd`: `0.9318x`
- chain: `0.9615x`

## Read

The crash is fixed for H200 productionish. This is still not a performance win:
`bwd_fwd` improves, but `bwd_bwd` loses more, so the full chain is about 3.9%
slower than baseline in this bounded run.

Next optimization work should focus on reducing `bwd_bwd` overhead without
returning to the `num_stages=2` shared-memory overflow.
