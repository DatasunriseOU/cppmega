# Mamba3 Stage2 Prod-Control - 2026-04-29

Branch: `worker/mamba3-stage2-prod-control`

Base: `worker/mamba3-stage2-force-nontma` at `972608d07a51b749a94c5568319435c2de551a24`

Goal: make the current best Mamba3 stage2 force-nonTMA mode
`bf_num_stages=1, bb_num_stages=0` easier to merge and test without changing
production defaults, then run a broader H200 A/B shape sweep against baseline.

## Production-Control Path

Added:

- `cppmega/megatron/upstream_patches/apply_mamba3_stage2_force_nontma_patches.py`

The applier is intentionally default-off. It mutates installed
`mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd.py` only when both gates are set:

```text
CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1
MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1
python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
```

The script writes a side-by-side backup before applying the existing patch:

- `mamba3_mimo_bwd.py.cppmega_stage2_force_nontma.bak`

Rollback guard:

```text
CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK=1
python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
```

If the backup is missing, rollback falls back to `patch -R`; if that also
fails, the script refuses and requires reinstalling `mamba_ssm`. The validation
guard also refuses a `bb_num_stages=1` patched default, because the production
candidate is the asymmetric `(bf=1,bb=0)` mode only.

Local checks:

```text
python -m py_compile \
  cppmega/megatron/upstream_patches/apply_mamba3_stage2_force_nontma_patches.py \
  scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  scripts/modal_mamba3_stage2_force_nontma_probe.py

PYTHONPATH=. python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
# SKIP CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA is not set

PYTHONPATH=. CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1 \
  python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
# FAIL: Refusing to mutate installed mamba_ssm without
# MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1
```

No production import path was changed in this branch.

## H200 Shape Sweep

Run:

```text
GHCR_TAG=785c3fd \
CPPMEGA_MODAL_GPU=H200:2 \
timeout 1800s \
modal run scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_prod_control_h200_bf1bb0_sweep_20260429_1 \
  --shape-csv smoke,representative,productionish \
  --variant-csv baseline,stage2_force_nontma \
  --warmup 2 \
  --iters 8
```

Modal app:

- `ap-E61edrZd9Lw70mZFZABTut`
- state after run: `stopped`, `Tasks=0`

Artifacts in Modal Volume `cppmega-mamba3-benchmarks`:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_prod_control_h200_bf1bb0_sweep_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_prod_control_h200_bf1bb0_sweep_20260429_1/summary.csv`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_prod_control_h200_bf1bb0_sweep_20260429_1/summary.json`

Device:

- GPU: `NVIDIA H200`
- device count: `2`
- capability: `(9, 0)`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

## Results

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | bwd_fwd WS/TMA | bwd_bwd WS/TMA | chain speedup | status |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| smoke | baseline | 0.0778 | 0.1624 | 0.2248 | no / 0 | no / 0 | 1.0000x | ok |
| smoke | stage2 `(1,0)` | 0.0803 | 0.1613 | 0.2267 | yes / 4 | no / 0 | 0.9916x | ok |
| representative | baseline | 0.2832 | 0.6704 | 0.9448 | no / 0 | no / 0 | 1.0000x | ok |
| representative | stage2 `(1,0)` | 0.2763 | 0.6626 | 0.9280 | yes / 4 | no / 0 | 1.0181x | ok |
| productionish | baseline | 1.8982 | 3.7266 | 5.6015 | no / 0 | no / 0 | 1.0000x | ok |
| productionish | stage2 `(1,0)` | 1.8103 | 3.7149 | 5.5006 | yes / 5 | no / 0 | 1.0183x | ok |

Correctness:

- `max_main_grad_abs_diff=0.0` for smoke, representative, and productionish.
- All tracked output diffs had `max_abs=0.0`, including `qk_dot` and `states`.

No TMA descriptor 716, misaligned-address, or dynamic-smem launch failure
occurred in this sweep.

H100 was not run in this pass. There were unrelated active Modal apps from
parallel work at the time of hygiene check, so I kept this task bounded to the
requested H200 production-control sweep and did not start another GPU class.

## Risk List

- The patch path still mutates installed `mamba_ssm` source. It is guarded and
  reversible, but env-off does not undo a prior file rewrite; rollback or
  reinstall is required.
- This is a TileLang source patch, not an upstreamed API. Upstream
  `mamba3_mimo_bwd.py` drift can make the patch fail or, worse, partially
  match. The applier validates sentinel markers and refuses partial states.
- The measured win is modest, about 1.8% chain on representative/productionish
  in this sweep. It is real enough for guarded A/B, but not enough to justify
  default-on production exposure.
- `bwd_bwd` WS/TMA remains a regression on H200. Do not merge variants that set
  `bb_num_stages > 0` as production candidates.
- NCU did not succeed in earlier Modal attempts, so the mechanistic attribution
  is still from TileLang source markers/timing, not counter-level profiling.

## Merge Recommendation

Merge only the production-control path and status docs, keeping the feature
default-off behind:

- `CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1`
- `MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1`

For production A/B, test only `stage2_force_nontma=(bf=1,bb=0)`. The broader
H200 sweep found no hidden correctness or productionish launch regression and
showed `1.0183x` chain speedup on the productionish shape. Do not make this the
production default until a longer end-to-end training A/B confirms the kernel
microbench win survives full workload variance.
