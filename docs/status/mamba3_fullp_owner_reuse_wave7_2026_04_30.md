# Mamba3 Full-P Owner Reuse Wave 7 - 2026-04-30

Status: evidence
Canonical: none
Date: 2026-04-30
Scope: move wave6 `G.T` reuse into the real TileLang stage2 `bwd_bwd` path.
Branch: `worker/mamba3-fused-m-tile-owner`

## Source Sites

The real TileLang `mamba_mimo_bwd_bwd_kernel` has the same two full
`F x F` surfaces found in the wave6 microbench, where `F = chunk_size * R`:

1. `dqk_from_diag_frag = dPhiO_shared @ PsiV_shared.T`
   - Source expression:
     `T.gemm(dPhiO_shared, PsiV_shared, dqk_from_diag_frag, transpose_B=True, clear_accum=True)`.
   - Consumers: `DGAMMA_DIAG`, then gamma-scaled diagonal DQ/DK.

2. `dk_intrachunk_frag = PsiV_shared @ dPhiO_shared.T`
   - Source expression:
     `T.gemm(PsiV_shared, dPhiO_shared, dk_intrachunk_frag, transpose_B=True, clear_accum=True)`.
   - Consumers: `DSSDA`, masked intrachunk DK, and masked intrachunk DQ.

These are exact transposes before the gamma scaling:

```text
dk_intrachunk_frag == dqk_from_diag_frag.T
```

## Patch

Added a guarded, non-default source patch:

- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_diag_reuse.patch`

The patch applies after
`mamba3_bwd_stage2_force_nontma.patch` and changes only stage2 `bwd_bwd`:

- allocate `dk_intrachunk_frag` immediately after `DGAMMA_DIAG`;
- fill it from raw `dqk_from_diag_frag[csr_j, csr_i]`;
- keep the existing gamma scaling/copy for `dqk_from_diag_shared`;
- remove the second `T.gemm(PsiV_shared, dPhiO_shared, ...)`.

Harness changes:

- added variant `stage2_bf1_bb0_diag_reuse`;
- added stage2-relative comparisons against `stage2_bf1_bb0`;
- made Modal app name configurable via `CPPMEGA_MODAL_APP_NAME`;
- made automatic upstream baseline insertion configurable via
  `CPPMEGA_STAGE2_AUTO_BASELINE=0`.

## Validation

Local:

```text
python -m py_compile scripts/modal_mamba3_stage2_force_nontma_benchmark.py
patch -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_stage2_force_nontma.patch
patch --dry-run -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_stage2_diag_reuse.patch
git diff --check
```

All passed.

## H200 Smoke

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s modal run \
  scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id mamba3_fullp_owner_reuse_wave7_smoke_frag_20260430_1 \
  --shape-csv smoke \
  --variant-csv baseline,stage2_bf1_bb0,stage2_bf1_bb0_diag_reuse \
  --warmup 1 --iters 3
```

App: `ap-tqmUt7x0ZS20BDrIdhclny`, stopped.

Shape: `B=1,S=256,H=4,N=64,P=64,R=4,chunk=16`.

| variant | bwd_fwd ms | bwd_bwd ms | chain ms |
| --- | ---: | ---: | ---: |
| baseline | 0.08037 | 0.16236 | 0.22401 |
| `stage2_bf1_bb0` | 0.08044 | 0.16059 | 0.22514 |
| `stage2_bf1_bb0_diag_reuse` | 0.08014 | 0.16484 | 0.22899 |

Correctness vs `stage2_bf1_bb0`:

- `max_main_grad_abs_diff = 0.0`
- all tracked outputs had `max_abs = 0.0` except `dssda = 8.67e-19`
- bwd_bwd same-run speed ratio: `0.9742x`

## H200 Productionish

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 \
CPPMEGA_MODAL_APP_NAME=cppmega-mamba3-stage2-reuse-wave7 \
CPPMEGA_STAGE2_AUTO_BASELINE=0 \
timeout 1500s modal run scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id mamba3_fullp_owner_reuse_wave7_productionish_stage2only_20260430_1 \
  --shape-csv productionish \
  --variant-csv stage2_bf1_bb0,stage2_bf1_bb0_diag_reuse \
  --warmup 2 --iters 8
```

App: `ap-JAAz3NVBFALGvbJaK7752t`, stopped.

Shape: `B=4,S=4096,H=32,N=64,P=128,R=4,chunk=16`.

| variant | bwd_fwd ms | bwd_bwd ms | chain ms |
| --- | ---: | ---: | ---: |
| `stage2_bf1_bb0` | 1.79777 | 3.71290 | 5.48303 |
| `stage2_bf1_bb0_diag_reuse` | 1.79160 | 3.74990 | 5.51330 |

Correctness vs `stage2_bf1_bb0`:

- `max_main_grad_abs_diff = 0.0`
- `qk_dot = 0.0`, `states = 0.0`
- all tracked outputs had `max_abs = 0.0` except `dssda = 3.47e-18`
- bwd_bwd same-run speed ratio: `0.9901x`
- chain same-run speed ratio: `0.9945x`

## Readout

This is correct within existing tolerances but does not survive as a production
patch. On the full stage2 productionish shape it regresses `bwd_bwd` by about
`1.0%` and chain time by about `0.55%`.

Likely reason: the removed tensor-core dot is replaced by an explicit
`F x F` fragment transpose and longer live range, and TileLang/Hopper handles
the original second GEMM efficiently enough that register pressure does not
drop into a better schedule.

Merge risk: high for performance, low for correctness. Keep the patch as
negative evidence / a patchable prototype, but do not enable it by default.

## Modal Cleanup

Apps launched for this wave:

- `ap-iVl2UVkU1men2l7i0EzsR8` - stopped (discarded shared-transpose smoke)
- `ap-tqmUt7x0ZS20BDrIdhclny` - stopped
- `ap-gLPwGn82ODBsbrVJSy8tzd` - stopped externally before results
- `ap-JAAz3NVBFALGvbJaK7752t` - stopped

Final `modal app list --json` showed the wave7 reuse app stopped with zero
tasks. One unrelated `cppmega-mamba3-stage2-profile-wave7` app was active, and
the pre-existing deployed `cppmega-prebuilt-smoke` app had zero tasks; neither
was launched or stopped by this work.
