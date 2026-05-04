# Mamba3 Stage2 Force Non-TMA Benchmark - 2026-04-29

Branch: `worker/mamba3-stage2-force-nontma`

Commit under test before this doc: `3a1dec3`

Goal: compare the stage2 force-nonTMA TileLang MIMO backward patch against the
baseline upstream TileLang kernels on H200, after smoke showed that targeted
`disable_tma=True` avoids the small float32 vector-slice TMA descriptor failure
while preserving WS on large TMA-capable paths.

## Harness

Script:

- `scripts/modal_mamba3_stage2_force_nontma_benchmark.py`

Compared variants:

- `baseline`: upstream non-TMA/non-WS kernels, `bf_num_stages=0`,
  `bb_num_stages=0`.
- `stage2_force_nontma`: applies
  `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_force_nontma.patch`,
  flattens Q/K and QK_DOT, sets `bf_num_stages=2`, `bb_num_stages=2`,
  and forces only the small vector-slice copies off TMA.

Run:

```text
GHCR_TAG=785c3fd \
CPPMEGA_MODAL_GPU=H200:2 \
timeout 900s \
modal run scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_force_nontma_h200_20260429_1 \
  --shape-csv representative,productionish \
  --warmup 2 \
  --iters 6
```

Modal app:

- `ap-rk8QiCBDcHLZ1ge0T95uxM`
- app name: `cppmega-mamba3-stage2-force-nontma-benchmark`
- state after run: stopped, `Tasks=0`

Artifacts in Modal Volume `cppmega-mamba3-benchmarks`:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_20260429_1/summary.csv`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_20260429_1/summary.json`

Device:

- GPU: `NVIDIA H200`
- device count: `2`
- capability: `(9, 0)`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

## Results

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | WS | TMA loads | status |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| representative | baseline | 0.2827 | 0.6685 | 0.9422 | no/no | 0/0 | ok |
| representative | stage2_force_nontma | 0.2956 | 0.7317 | 1.0008 | yes/yes | 4/6 | ok |
| productionish | baseline | 1.8790 | 3.7216 | 5.5642 | no/no | 0/0 | ok |
| productionish | stage2_force_nontma | n/a | n/a | n/a | n/a | n/a | crashed |

Representative correctness against baseline:

- `max_main_grad_abs_diff`: `0.0`
- all tracked output diffs: `max_abs=0.0`

Representative speed:

- `bwd_fwd` speedup ratio: `0.9564`
- `bwd_bwd` speedup ratio: `0.9137`
- chain speedup ratio: `0.9415`

Productionish failure:

```text
InternalError: Failed to set the allowed dynamic shared memory size to 231712
```

## Read

This is not a production candidate yet.

The patch does what it was designed to do mechanically: the small vector-slice
copies no longer trigger the descriptor 716 misaligned TMA failure, and WS still
fires on the large TMA-capable paths. But on H200 it is slower on the
representative shape and exceeds dynamic shared memory limits on the
productionish shape.

The useful part to keep is the precise per-copy non-TMA approach. The current
`bf_num_stages=2`/`bb_num_stages=2` WS variant should not replace baseline until
shared memory is reduced and a production shape beats baseline.
