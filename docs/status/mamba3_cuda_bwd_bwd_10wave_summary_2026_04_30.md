# Mamba3 CUDA bwd_bwd 10-Wave Summary - 2026-04-30

Worktree: `/home/dave/source/cppmega/.claude/worktrees/mamba3-cuda-full-bwd-ab`

Branch: `worker/mamba3-cuda-full-bwd-ab`

Scope: final Lane C A/B and readiness decision artifact for the 10-wave
Mamba3 `mamba_mimo_bwd_bwd` campaign. This report uses completed wave data
through Wave 9 plus the current best-known Wave 10 placeholder number. It does
not wait for any live Lane A/B reruns.

## Decision

Ship only the guarded TileLang stage2 candidate:

```text
bf_num_stages = 1
bb_num_stages = 0
```

Do not replace production `mamba_mimo_bwd_bwd` with the CUDA warp-owner path in
this campaign.

The stage2 `(bf=1,bb=0)` path is mergeable because it is exact against the
baseline outputs and has a repeated H200 productionish chain win in the
1.6-1.9% range:

| source | baseline chain ms | stage2 `(bf=1,bb=0)` chain ms | speedup | correctness |
| --- | ---: | ---: | ---: | --- |
| profile matrix | 5.5525 | 5.4667 | 1.0157x | `max_main_grad_abs_diff=0.0`, `qk_dot/states=0.0` |
| longer default confirmation | 5.5628 | 5.4567 | 1.0194x | all tracked diffs `0.0` |
| Wave 8 AB refresh | 5.59245 | 5.50098 | 1.0166x | stage2 vs baseline main grad diff `0.0` |

Read: this is not a large kernel breakthrough, but it is the only completed
candidate with production-shaped correctness, bounded performance upside, and a
small enough integration surface to merge behind a guard.

## Current Best Candidate State

### Mergeable Production Candidate

`stage2_bf1_bb0` keeps the useful half of the stage2 force-nonTMA work:

- `bwd_fwd`: flattened Q/K and QK_DOT with TMA/WS enabled.
- `bwd_bwd`: remains on the non-WS/non-TMA path.

The asymmetric result matters. The `(bf=0,bb=1)` and old `(1,1)` forms proved
that enabling WS/TMA inside `bwd_bwd` hurts H200 productionish:

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | chain speedup |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.8740 | 3.7103 | 5.5525 | 1.0000x |
| `stage2_bf1_bb0` | 1.8063 | 3.7097 | 5.4667 | 1.0157x |
| `stage2_bf0_bb1` | 1.9665 | 3.9734 | 5.9092 | 0.9396x |
| old `(1,1)` | 1.7919 | 3.9727 | 5.7331 | 0.9685x |

Production recommendation:

1. Merge the guarded `(bf=1,bb=0)` default.
2. Keep all older stage2 modes available only as explicit benchmark variants.
3. Do not enable `bb_num_stages > 0` for production H200 `bwd_bwd`.

### CUDA Warp-Owner Path

The custom CUDA path is promising at the component-economics level, but not
ready for production replacement.

Best current read:

| item | ms | note |
| --- | ---: | --- |
| TileLang stage2 `bwd_bwd` reference | 3.70674 | Wave 6/7 H200 productionish reference |
| Wave 9 covered CUDA subset | 2.48042 | current best-known covered subset |
| ratio | 0.6692 | subset / TileLang |
| speedup floor | 1.4944x | TileLang / subset |
| remaining `bwd_bwd` budget before parity | 1.22632 | for all missing work |

The covered CUDA subset includes:

- `DGAMMA_DIAG`;
- same-time diagonal contributions into `DK` and `DQ`;
- same-time `qk_dot -> dPsiV -> DV`;
- qk-dot contribution into `DMIMO_V` through the output-owner all-R sidecar.

Receipts behind that read:

| source | component | ms |
| --- | --- | ---: |
| Wave 7 | diag + qk/dV one-launch slice | 1.91459 |
| Wave 8 AB refresh | diag + qk/dV one-launch slice | 1.92990 |
| Wave 8 sidecar | qk-`DMIMO_V` output-owner all-R | 0.53634 |
| Wave 9 canonical projection | Wave 7 + sidecar | 2.45093 |
| Wave 9 AB-normalized projection | Wave 8 AB + sidecar | 2.46624 |
| Wave 10 placeholder/current best-known | covered subset | 2.48042 |

This path cannot replace production yet because it is missing full output
parity at the `mamba_mimo_bwd_bwd` boundary. The missing set is not cosmetic:

- off-time intra-chunk/state work;
- full `DK`, `DQ`, `DV`, and `DMIMO_V`, not only same-time slices;
- scalar outputs: `dfactor`, `dangles`, `dd`, `dda`, `dssda`,
  `dda_cs_rev`, and `dda_cs`;
- state/LKQ/D tensorization inside the candidate rather than scalar or Python
  epilogue leftovers;
- memory measurement in the integrated autograd lifetime;
- end-to-end training A/B.

Memory caveat: Wave 8 AB's standalone CUDA component harness peaked at
`6.92774 GiB` allocated versus TileLang stage2 at `4.73024 GiB`. That standalone
delta is not a production memory claim because it keeps references,
duplicated outputs, sidecar temporaries, and independent component inputs alive
together. A production replacement must show integrated peak memory at or below
the TileLang stage2 path. The output-owner all-R `DMIMO_V` direction is the
only acceptable sidecar memory direction so far because it uses no partial
tensor and no atomics.

## Closed Paths

These paths should not consume more Wave 10 time.

| path | status | evidence/read |
| --- | --- | --- |
| ParaRNN / Apple for M2RNN | closed for exact dense M2RNN | M2RNN is mathematically compatible with ParaRNN, but Apple kernels target diagonal or tiny block cases. Exact `V=16` dense Jacobian work would carry about 256 fp32 scalars before temporaries and likely collapse occupancy. Random probes needed six Newton iterations, not the small fixed count that would make this attractive. |
| P_TILE-only `DMIMO_V` output ownership | closed/superseded | The useful Wave 9 sidecar is output-owner all-R. P-tile-only ownership does not solve the broader ownership split and is not the production direction. |
| row/split and split/post kernels | closed | Wave 1, Wave 2, and Wave 5 showed correct split/post variants but full-chain regressions. Productionish Wave 5 split CUDA was `6.53346 ms` `bwd_bwd` versus stage2 `3.69708 ms`. |
| TileLang in-body R x R / `G.T` reuse ideas | closed as written | Current TileLang expressions either serialize over `P`, recreate the full padded GEMM work, or need large accumulators inside an already heavy `bwd_bwd` body. The full-kernel TileLang R x R patch was correct on smoke but regressed `bwd_bwd` from `0.1635 ms` to `0.5519 ms`. |
| scalar state/LKQ/D epilogue prototypes | closed for replacement | These can be correctness scaffolding only. Production requires state, LKQ, D, and scalar-gradient outputs tensorized inside the replacement boundary with no Python epilogue or scalar fallback on the hot path. |
| NCU on Modal | closed for this decision | The AB harness intentionally uses CUDA events, generated-source metadata, and `torch.cuda` memory counters. Modal NCU is not required for readiness and should not block the decision. If a focused Nsight Compute report is needed, run it on a controlled local/bench host, not as a Modal gating step. |

## What Must Happen Before CUDA Replacement

A CUDA replacement can reopen only after all of the following are true:

1. Full output parity against TileLang at the real `mamba_mimo_bwd_bwd` call
   boundary, for every output tensor.
2. State, LKQ, D, and scalar-gradient tensorization inside the candidate.
3. Full off-time/state work implemented, not stubbed or copied from TileLang.
4. Full `DK/DQ/DV/DMIMO_V` accumulation implemented with a production ownership
   model.
5. Integrated memory peak at or below TileLang stage2, measured in the real
   autograd lifetime.
6. Launch count justified by measured chain speedup; one `bwd_bwd` replacement
   launch remains the target.
7. H200 productionish correctness and timing rerun.
8. H100 smoke or representative portability rerun.
9. Training A/B against the guarded stage2 candidate, not just component
   microbenchmarks.

Until those gates pass, the CUDA path remains an R&D candidate and the
production candidate remains guarded stage2 `(bf=1,bb=0)`.

## Modal App Hygiene

Observed hygiene from the wave docs:

- wave-owned Modal apps were bounded with `timeout`;
- completed wave apps were stopped and showed `Tasks=0`;
- pre-existing deployed apps such as `cppmega-prebuilt` were left alone unless
  explicitly wave-owned;
- Wave 8/9 AB used CUDA events and memory counters instead of NCU.

Rules for any Wave 10 update run:

1. Use a unique `--run-id`.
2. Run at most one app per GPU class at a time.
3. After collecting JSON artifacts, check `modal app list`.
4. Stop only wave-owned running apps.
5. Leave unrelated pre-existing deployed apps untouched.

## Optional Wave 10 Update Commands

These commands are placeholders for refreshing numbers. They are not required
for this final decision.

Stage2 final confirmation:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 900s \
  modal run scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_force_nontma_h200_wave10_final_20260430_1 \
  --shape-csv productionish \
  --variant-csv baseline,stage2_force_nontma \
  --warmup 2 \
  --iters 12
```

Full AB harness refresh:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 1800s \
  modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
  --run-id wave10_h200_ab_update_20260430_1 \
  --shape-csv smoke,productionish \
  --iters 6 \
  --warmup 2 \
  --cuda-iters 10 \
  --cuda-warmup 3
```

`DMIMO_V` sidecar refresh:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 1200s \
  modal run scripts/modal_mamba3_dmimo_reduce_wave8_cuda.py \
  --shape-csv productionish \
  --iters 20 \
  --warmup 5
```

Cheap H100 smoke:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100:2 timeout 900s \
  modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
  --run-id wave10_h100_smoke_20260430_1 \
  --shape-csv smoke \
  --iters 2 \
  --warmup 1 \
  --cuda-iters 5 \
  --cuda-warmup 1
```

Modal cleanup check:

```bash
modal app list
modal app stop <wave-owned-app-id-or-name>
```

## Final Verdict

Merge guarded stage2 `(bf=1,bb=0)` as the production candidate. It is exact and
has a repeatable small H200 productionish chain win.

Do not ship the CUDA warp-owner path as a replacement. Its covered subset is
fast enough to justify future R&D (`2.48042 ms` versus `3.70674 ms` TileLang
`bwd_bwd`), but it is missing the hard parts of the production contract:
state/off-time math, scalar outputs, full tensor parity, and integrated memory.
