# Mamba3 R x R Diagonal CUDA Integration Wave 6 - 2026-04-29

Branch: `worker/mamba3-rr-diag-cuda-integrate`

## Goal

Move past the wave5 host post-kernel split and build a meaningful in-launch
CUDA prototype for the `bwd_bwd` same-time `R x R` diagonal consumers.

Prior wave5 productionish reference:

| path | bwd_bwd ms | chain ms |
| --- | ---: | ---: |
| `stage2_bf1_bb0` | 3.6971 | 5.4528 |
| `stage2_rr_diag_cuda` split | 6.5335 | 8.2905 |

## Inspected Artifacts

Current TileLang/H200 stage2 baseline run:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1500s \
modal run --timestamps scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_wave6_baseline_20260429_1 \
  --shape-csv smoke,productionish \
  --variant-csv baseline,stage2_bf1_bb0 \
  --warmup 1 \
  --iters 4
```

Artifacts in Modal Volume `cppmega-mamba3-benchmarks`:

- `/mamba3_stage2_force_nontma_benchmark/stage2_wave6_baseline_20260429_1/report.json`
- `/mamba3_stage2_force_nontma_benchmark/stage2_wave6_baseline_20260429_1/productionish/stage2_bf1_bb0/bwd_bwd_kernel_source.cu`
- `/mamba3_stage2_force_nontma_benchmark/stage2_wave6_baseline_20260429_1/productionish/stage2_bf1_bb0/bwd_fwd_kernel_source.cu`

TileLang version: `0.1.8+cu132.gitf309d814`.

Productionish `stage2_bf1_bb0` generated source markers:

| source | chars | sha256 | launch bounds | TMA loads/stores | WS |
| --- | ---: | --- | --- | ---: | --- |
| `bwd_fwd_kernel_source.cu` | 43801 | `12e7426c...6274d5d7` | `(256, 1)` | 5 / 3 | yes |
| `bwd_bwd_kernel_source.cu` | 83354 | `63da45df...79618bab` | `(256, 1)` | 0 / 0 | no |

The wave4/wave5 CUDA artifact was also inspected:

- `rr_diag_cuda_kernel.cu` had the wave4 standalone timestep CTA kernel and
  the wave5 `stage2_rr_diag_post_kernel`;
- wave5 post path still launches after TileLang `bwd_bwd`, rereads `DOUT/V`,
  `Q/K`, and already-stored DK/DQ, then adds and stores again.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave6_inlaunch_cuda.py`
- `scripts/modal_mamba3_rr_diag_wave6_inlaunch_cuda.py`

Two in-launch slice prototypes were added, both using production stage2 tensor
layouts:

1. `stage2_rr_diag_chunk_owner_kernel`
   - one CTA owns one `(B, H, chunk)` tile;
   - loops over all 16 timesteps in the chunk;
   - computes `dPhiO @ PsiV.T` same-time `4 x 4` blocks and writes
     `DGAMMA_DIAG`, `DK` diagonal delta, and `DQ` diagonal delta directly.

2. `stage2_rr_diag_chunk_warp_owner_kernel`
   - still one CTA per `(B, H, chunk)`;
   - four warps compute four timesteps at a time;
   - uses warp reductions, no dynamic shared memory;
   - this is the useful wave6 prototype.

Both avoid the wave5 split envelope: there is no extra reload/add/store of
TileLang DK/DQ outputs in the measured kernel.

## Local Checks

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave6_inlaunch_cuda.py \
  scripts/modal_mamba3_rr_diag_wave6_inlaunch_cuda.py

git diff --check

python upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave6_inlaunch_cuda.py \
  --shape smoke --device cpu --iters 1 --warmup 0
```

All passed.

## H200 Runs

Wave6 slice command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 1200s \
modal run --timestamps scripts/modal_mamba3_rr_diag_wave6_inlaunch_cuda.py \
  --shape-csv smoke,productionish \
  --warmup 3 \
  --iters 10
```

Device:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

## Correctness

Reference is the wave5 production-layout CUDA timestep post slice, run into
zeroed DK/DQ outputs. This validates the same diagonal math and stage2 layout.

| shape | variant | dgamma max abs | DK delta max abs | DQ delta max abs |
| --- | --- | ---: | ---: | ---: |
| smoke | chunk owner | 0.0 | 0.0 | 0.0 |
| smoke | chunk warp owner | 1.776e-15 | 1.137e-13 | 2.274e-13 |
| productionish | chunk owner | 0.0 | 0.0 | 0.0 |
| productionish | chunk warp owner | 6.217e-15 | 4.547e-13 | 4.547e-13 |

The stage2 `stage2_bf1_bb0` outputs remained exactly equal to baseline in the
full harness: max main grad diff `0.0` on both smoke and productionish.

## Performance

Full stage2 H200 refresh:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms |
| --- | --- | ---: | ---: | ---: |
| smoke | baseline | 0.08255 | 0.16770 | 0.23243 |
| smoke | `stage2_bf1_bb0` | 0.08552 | 0.16666 | 0.23474 |
| productionish | baseline | 1.87771 | 3.71713 | 5.57603 |
| productionish | `stage2_bf1_bb0` | 1.80246 | 3.70674 | 5.47613 |

Diagonal slice timings:

| shape | slice | mean ms | read |
| --- | --- | ---: | --- |
| smoke | wave5 timestep post CUDA slice | 0.01682 | extra launch shape, timestep CTA |
| smoke | wave6 chunk owner | 0.06875 | too serial; only 64 CTAs |
| smoke | wave6 chunk warp owner | 0.02500 | closer, but tiny shape underfilled |
| productionish | wave5 timestep post CUDA slice | 3.16204 | production layout, but split-style ownership |
| productionish | wave6 chunk owner | 3.09658 | barely faster; serial chunk loop |
| productionish | wave6 chunk warp owner | 1.77566 | useful in-launch signal |

Productionish comparisons for the useful warp-owner variant:

- `1.78x` faster than the production-layout timestep CUDA slice
  (`3.1620 / 1.7757`);
- `0.864x` the old wave4 standalone CUDA diagonal time (`1.7757 / 2.0560`);
- `48.0%` of the refreshed full `stage2_bf1_bb0` `bwd_bwd` time
  (`1.7757 / 3.7067`).

## Resource Metadata

| kernel | regs/thread | dyn smem | active blocks/SM | occupancy |
| --- | ---: | ---: | ---: | ---: |
| wave5 timestep post CUDA | 48 | 8256 B | 10 | 62.5% |
| wave6 serial chunk owner | 72 | 8256 B | 7 | 43.75% |
| wave6 warp chunk owner | 88 | 0 B | 5 | 31.25% |

The warp-owner kernel wins despite lower theoretical occupancy because it
preserves enough timestep parallelism inside the chunk CTA and avoids shared
memory reductions.

## Blockers

This is not yet a full `bwd_bwd` replacement. Missing integration contract:

- a stable way to call a CUDA device helper from the generated TileLang kernel
  at the exact point where `dPhiO`, `PsiV`, `q_pre_rot`, `k_pre_rot`,
  `qk_dot`, and gamma-equivalent values are already live;
- ownership of the surrounding `bwd_bwd` work: non-diagonal DK/DQ path, DV,
  `DFactor`, `DA`, angle grads, state-loop effects, and output ordering;
- a resource plan for a full CUDA/CuTe chunk kernel. The useful slice already
  costs 88 regs/thread, so directly adding the rest of `bwd_bwd` around it
  will need deliberate register staging or split subpasses.

TileLang in-body patching is still unattractive: the previous diagonal-only
TileLang patch serialized over `P`, while external CUDA post-kernels pay the
launch and store/reload boundary that wave5 measured.

## Read

The CUDA in-launch path survives as a full/custom `bwd_bwd` rewrite candidate.
It does not survive as another post-kernel split.

The strongest wave6 datapoint is the warp-owner chunk CTA: it computes the
same production-layout diagonal consumers in `1.7757 ms` on the productionish
shape, faster than both the wave5 production-layout timestep slice and the old
wave4 standalone diagonal microbench. However, it is only a slice and already
accounts for roughly half of refreshed full `stage2_bf1_bb0 bwd_bwd`, so a
successful next wave must replace surrounding work in the same launch rather
than add this beside TileLang.

## Next Wave Recommendation

Do not spend another wave on split/post diagonal paths.

Next wave should build a monolithic CUDA/CuTe chunk-level `bwd_bwd` skeleton
around `stage2_rr_diag_chunk_warp_owner_kernel`:

1. port enough surrounding DK/DQ/DGAMMA work to keep `dPhiO`, `PsiV`,
   `q_pre_rot`, `k_pre_rot`, and gamma resident;
2. initially target DK/DQ/DGAMMA parity only, with DV/DA/angle outputs either
   stubbed or copied from a reference path;
3. then decide whether the register budget can fit a one-launch full chunk or
   needs a two-subpass CUDA design.

Keep `stage2_bf1_bb0` as the production baseline until that larger in-launch
CUDA path exists.

## Modal Cleanup

Apps started or touched in this wave:

- `ap-OGMiRdEIGVRxdrVWzJ1MU2`: completed/stopped, tasks=0.
- `ap-rk4uZGJFA8BNTaPsDnnPvg`: completed/stopped, tasks=0.
- `ap-O9MdpYGTjXjUVWlZHwQUJG`: completed/stopped, tasks=0.
- `ap-uktsnlfQ2u74bnoiMC0ILt`: explicitly stopped after it showed one running
  task, tasks=0 after stop.
- `ap-pFgKQgpDW38GlDJd84C8I0`: explicitly stopped after the final app-list
  check showed one running task, tasks=0 after stop.

Pre-existing deployed app `cppmega-prebuilt` had tasks=0 and was left alone.
