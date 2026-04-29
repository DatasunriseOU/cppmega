# Mamba3 bwd_bwd on-chip reuse - 2026-04-29

Branch: `worker/mamba3-bwd-bwd-onchip-reuse`

Base: `worker/mamba3-stage2-force-nontma` / `972608d`

Goal: reduce the `bwd_bwd` live set around `PsiV/Psi/v/qk` without adding a
global `[B,H,S*R,P]` cache and without relying on bwd_bwd WS/TMA.

## Live Region Inspected

Hot region in `mamba_mimo_bwd_bwd_kernel`:

- `dPhiO_shared` and `dstates_shared` are live through the reverse chunk body.
- `dPsiV_frag` accumulates interchunk + intrachunk terms, then
  `dPsiV_D_fused_frag` adds D/qk diagonal contributions and is copied to
  `dPsiV_combined_shared`.
- `v_frag` and `Psi_frag` are then used for `dv`, `dPsi_acc`, and `PsiV`.
- Baseline builds `PsiV_frag = [chunk_size, R, P]`, then copies it into
  `PsiV_shared = [fused_chunk_size, P]` for `dqk_from_diag`, `dk_frag`, and
  `dk_intrachunk_frag` GEMMs.
- `qk_dot_shared` is already the stage2 flattened shared path; no bwd_bwd WS/TMA
  was enabled.

## Variants

Patch chain for both candidates:

1. `mamba3_bwd_stage2_force_nontma.patch`
2. candidate on-chip patch

Candidates:

- `mamba3_bwd_bwd_onchip_psiv_direct.patch`
  - removes `PsiV_frag`;
  - writes `PsiV_shared[cs*R+r, p] = v_frag[cs, p] * Psi_frag[r, p]` directly.
- `mamba3_bwd_bwd_onchip_psiv_direct_dpsi_acc.patch`
  - includes `psiv_direct`;
  - also removes the `dPsi_frag` copy-through and updates `dPsi_acc` directly.

## Local Validation

```text
patch -p4 mamba3_mimo_bwd.py < mamba3_bwd_stage2_force_nontma.patch
patch --dry-run -p4 mamba3_mimo_bwd.py < mamba3_bwd_bwd_onchip_psiv_direct.patch
patch --dry-run -p4 mamba3_mimo_bwd.py < mamba3_bwd_bwd_onchip_psiv_direct_dpsi_acc.patch
python -m py_compile scripts/modal_mamba3_bwd_bwd_onchip_reuse.py
python -m py_compile scripts/modal_mamba3_stage2_force_nontma_benchmark.py
```

All passed.

## H200 Smoke

Run:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run scripts/modal_mamba3_bwd_bwd_onchip_reuse.py \
  --run-id bwd_bwd_onchip_h200_smoke_20260429_2 \
  --shape-csv smoke \
  --variant-csv baseline,stage2_force_nontma,onchip_psiv_direct,onchip_psiv_direct_dpsi_acc \
  --warmup 1 --iters 2
```

Artifacts:

- `/benchmarks/mamba3_bwd_bwd_onchip_reuse/bwd_bwd_onchip_h200_smoke_20260429_2/report.json`
- `/benchmarks/mamba3_bwd_bwd_onchip_reuse/bwd_bwd_onchip_h200_smoke_20260429_2/summary.csv`

Smoke shape: `B=1, S=256, H=4, G=1, N=64, P=64, R=4`.

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | max main grad diff | bwd_bwd WS/TMA |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.0830 | 0.1666 | 0.2297 | n/a | no / 0 |
| stage2_force_nontma | 0.0868 | 0.1653 | 0.2326 | 0.0 | no / 0 |
| onchip_psiv_direct | 0.0831 | 0.1668 | 0.2336 | 0.0 | no / 0 |
| onchip_psiv_direct_dpsi_acc | 0.0850 | 0.1659 | 0.2315 | 0.0 | no / 0 |

## H200 Productionish

Run:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run scripts/modal_mamba3_bwd_bwd_onchip_reuse.py \
  --run-id bwd_bwd_onchip_h200_productionish_20260429_1 \
  --shape-csv productionish \
  --variant-csv baseline,stage2_force_nontma,onchip_psiv_direct,onchip_psiv_direct_dpsi_acc \
  --warmup 1 --iters 4
```

Artifacts:

- `/benchmarks/mamba3_bwd_bwd_onchip_reuse/bwd_bwd_onchip_h200_productionish_20260429_1/report.json`
- `/benchmarks/mamba3_bwd_bwd_onchip_reuse/bwd_bwd_onchip_h200_productionish_20260429_1/summary.csv`

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4`.

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | max main grad diff | bwd_bwd WS/TMA |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 1.8852 | 3.7168 | 5.5585 | n/a | no / 0 |
| stage2_force_nontma | 1.7890 | 3.6989 | 5.4586 | 0.0 | no / 0 |
| onchip_psiv_direct | 1.7885 | 3.6965 | 5.4669 | 0.0 | no / 0 |
| onchip_psiv_direct_dpsi_acc | 1.7908 | 3.6986 | 5.4676 | 0.0 | no / 0 |

## Read

The direct `PsiV_shared` write is semantically clean and compiles, but it is not
a production win. On productionish H200, `onchip_psiv_direct` improved bwd_bwd
only by about `0.0024ms` versus stage2 (`3.6965ms` vs `3.6989ms`), while the full
chain regressed (`5.4669ms` vs `5.4586ms`). The `dPsi_acc` direct-update variant
was neutral/slightly worse.

Recommendation: do not ship these variants as the default. Keep the patch files
as negative/neutral probes; the next useful work should target a larger live-set
reduction than deleting only `PsiV_frag`, likely around the `dPsiV`/`dqk` and
`dstates_shared` overlap, still without a global PsiV cache.
