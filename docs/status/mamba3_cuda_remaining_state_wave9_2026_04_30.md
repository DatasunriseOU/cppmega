# Mamba3 CUDA Remaining State/LKQ/D Wave 9 - 2026-04-30

Status: evidence
Canonical: none
Date: 2026-04-30
Scope: Wave9 measurement and ownership estimate for remaining Mamba3 CUDA `bwd_bwd` state/LKQ/D work.

Branch: `worker/mamba3-cuda-dmimo-reduce`

Base context:

- Wave7: `DGAMMA_DIAG` + qk diagonal `DK/DQ` + qk `DV`, measured
  `1.91459 ms` productionish.
- Wave8: qk `DMIMO_V` output-owner all-R, measured `0.53634 ms`.
- Current wave8 target before this lane's remaining work: `2.45093 ms`.
- TileLang `stage2_bf1_bb0` full `bwd_bwd`: `3.70674 ms`.

## Source/Algebra Inspection

Local `mamba_ssm.ops.tilelang` is not importable in this worktree, so I used:

- `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py`
- `cppmega/megatron/cute_dsl_mimo/test_all14_vs_tilelang.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_layout_fix.patch`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_bwd_rr_diag_tilelang.patch`

Remaining non-qk/state/LKQ/D work after wave7/8:

| output/path | remaining producer |
| --- | --- |
| `DV` | state `K @ dstates`, LKQ `masked(K @ Q.T) @ dPhiO`, direct `D * dPhiO` |
| `DMIMO_V` | same state/LKQ/D `dPsiV` producer, reduced over chunks/time |
| `DD` | scalar `sum(dPhiO)` when `D` is present |
| `DDA_CS_REV` | `(K * (PsiV @ dstates.T)).sum` |
| `DFACTOR` | `(K_pre_trap * dk_nodiag).sum` |
| `DSSDA` | `lkq_save * (PsiV @ dPhiO.T)` reduced to `[cs, cs]` |
| `DDA` | state-passing term from cached forward states and loop-carried `dstates` |
| `DDA_CS` | `(Q * (dPhiO @ states.T)).sum` |
| `DANGLES` | inverse rotary contributions from final `DK/DQ` |
| complete `DK/DQ` | state path, intra-chunk `PsiV @ dPhiO.T` path, trap scaling, inverse rotary, plus qk diagonal already prototyped |

The largest missing `dPsiV` producer is state+LKQ+D. It is also the ownership
boundary for non-qk `DMIMO_V`.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/state_lkq_d_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/state_lkq_d_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave9_state_lkq_d_cuda.py`
- `scripts/modal_mamba3_state_lkq_d_wave9_cuda.py`

CUDA probe:

- one CTA owns one `(B, H, chunk)` tile;
- materializes only an in-block `LKQ[64,64]` tile;
- computes state+LKQ+D `dPsiV` and writes `DV` and `DD` directly;
- optional path writes per-chunk `DMIMO_V` partials `[B,H,nchunks,R,P]`;
- final reducer writes `DMIMO_V[B,H,R,P]`.

This is a correctness/ownership skeleton using scalar CUDA loops, not the final
tensor-core implementation.

## H200 Runs

Smoke:

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_state_lkq_d_wave9_cuda.py \
  --shape-csv smoke --warmup 1 --iters 1 \
  --ref-iters 0 --skip-reference-timing
```

Productionish correctness/timing:

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_state_lkq_d_wave9_cuda.py \
  --shape-csv productionish --warmup 3 --iters 10 \
  --ref-iters 0 --skip-reference-timing
```

Productionish torch/reference cost-model timing:

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_state_lkq_d_wave9_cuda.py \
  --shape-csv productionish --warmup 1 --iters 3 \
  --ref-warmup 0 --ref-iters 1 --skip-reference
```

Device/runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

## Correctness

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| check | max abs diff |
| --- | ---: |
| `DV` vs torch reference | `4.768e-07` |
| `DD` vs torch reference | `1.335e-05` |
| `DV` with partials vs torch reference | `4.768e-07` |
| `DD` with partials vs torch reference | `1.526e-05` |
| `DMIMO_V` partial+reduce vs torch reference | `1.749e-07` |
| `DV` no-partial vs partial kernel | `0.000e+00` |

Smoke also passed; worst smoke diff was `DMIMO_V=5.322e-04`, from small-shape
bf16 accumulation order.

## Productionish Timings

| component | mean ms | notes |
| --- | ---: | --- |
| state/LKQ/D `DV+DD` chunk-owner skeleton | `29.84713` | scalar loops; no global `dPsiV` temp |
| state/LKQ/D `DV+DD+DMIMO_V partials` skeleton | `27.02475` | scalar loops; writes 64 MiB partial tensor |
| `DMIMO_V` partial final reduce | `0.03680` | same order as wave8 reducer |
| state/LKQ/D two-pass total | `27.05544` | skeleton + final reduce |
| torch/BMM reference cost model | `588.001` | allocation-heavy; not an implementation candidate |

Resource metadata:

| kernel | regs/thread | static smem | active blocks/SM | occupancy |
| --- | ---: | ---: | ---: | ---: |
| `DV+DD` skeleton | 64 | 16,896 B | 8 | 50.0% |
| `DV+DD+partials` skeleton | 56 | 18,944 B | 9 | 56.25% |
| partial reducer | 32 | 0 B | 16 | 100.0% |

Operation model for productionish:

- chunks: `32,768`
- causal LKQ entries/chunk: `1,920`
- state FMA: `17.18B`
- LKQ FMA: `4.03B`
- LKQ apply FMA: `8.05B`
- total state+LKQ producer FMA: `29.26B`
- piggybacked `DMIMO_V` partial tensor: `64.0 MiB`
- partial read+write+output traffic: `128.25 MiB`

## Ownership Read

For state/LKQ/D, a no-temp output-owner `DMIMO_V` kernel is the wrong owner:
it would recompute `K @ Q.T` and `K @ dstates` for each `(R, P)` output tile.
The wave8 qk output-owner worked because qk `dPsiV` is cheap per output; LKQ is
not.

The viable non-qk `DMIMO_V` ownership is chunk-owner while `dPsiV` is already
live, writing `[B,H,nchunks,R,P]` partials, followed by the cheap final reducer.
That reintroduces a 64 MiB temporary, but the final reduce is only `0.037 ms`;
the real cost is producing `dPsiV`, which must be tensor-core fused with the
rest of `bwd_bwd`.

## Projection

The scalar CUDA skeleton is not a candidate: adding it to wave8 projects to
`29.51 ms`, far slower than TileLang.

For a real CUDA/CuTe kernel, the relevant productionish work size is about
`29.26B` FMA for the state+LKQ producer measured here. The remaining
state/intra `DK/DQ` and scalar paths add roughly another `80-90B` FMA-equivalent
from `PsiV @ dstates`, `PsiV @ dPhiO.T`, `dki @ Q`, `dPhiO @ states.T`,
`dki.T @ K`, and `Q.T @ dPhiO`.

Full CUDA still likely beats TileLang if the remaining work is implemented as
tensor-core chunk kernels and fused around the existing chunk owner. A realistic
target is now close rather than automatic:

- wave8 baseline: `2.45093 ms`;
- optimized state/LKQ/D `DV/DMIMO/DD`: estimate `0.3-0.7 ms`;
- optimized remaining `DK/DQ` + scalar/state paths: estimate `0.6-1.2 ms`;
- projected full CUDA: `3.35-4.35 ms`, with the lower half beating TileLang
  `3.70674 ms`.

So the answer is: **yes, likely, but only with tensor-core fused state/intra
paths; scalar CUDA or standalone output-owner recomputation loses decisively.**
