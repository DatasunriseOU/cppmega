# Mamba3 bwd_bwd pressure reduction variants

Date: 2026-04-29

Branch: `worker/mamba3-bwd-bwd-pressure-reduce`
Base: `worker/mamba3-stage2-force-nontma` / `972608d`

## Scope

Task C looked for cheap TileLang source-level `bwd_bwd` register/shared-memory/live-set reductions without enabling `bwd_bwd` WS/TMA.  The baseline for new variants is the existing `stage2_force_nontma` patch: `bf_num_stages=1`, `bb_num_stages=0`, flattened Q/K and QK_DOT, `bwd_fwd` WS/TMA enabled, `bwd_bwd` non-WS.

Inspected pressure points in `mamba_mimo_bwd_bwd_kernel`:

- `q_frag` / `k_frag` full `[fused_chunk_size, N]` fragments used only to add Q/K bias before copying back to shared.
- Q/K rotary path with two `[chunk_size, R, N//rotary_dim_divisor]` half fragments for each of Q and K.
- QK_DOT path now uses `qk_dot_shared` directly for `dPsiV_D_fused_frag`, avoiding the old `qk_dot_frag` copy.
- `dgamma_diag_prereduce_frag [chunk_size, R*R]` copies `qk_dot_shared`, multiplies by `dqk_from_diag_shared`, then reduces.
- `dPhiO_shared`, `dstates_shared`, `PsiV_shared`, `states_shared`, and `dqk_from_diag_frag/shared` dominate the later live set; no large safe split was attempted.

## Variants

Patch files:

- `mamba3_bwd_bwd_pressure_reduce_direct_bias.patch`
  Applies Q/K bias directly in shared memory and removes the full `q_frag` / `k_frag` round trips in `bwd_bwd`.
- `mamba3_bwd_bwd_pressure_reduce_rotary_onetmp.patch`
  Replaces Q/K first+second rotary half fragments with one temp fragment plus extra shared-memory passes.
- `mamba3_bwd_bwd_pressure_reduce_dgamma_serial.patch`
  Removes `dgamma_diag_prereduce_frag` and computes `DGAMMA_DIAG` by a tiny per-step serial R x R accumulation.

Modal harness:

- `scripts/modal_mamba3_bwd_bwd_pressure_reduce_benchmark.py`
  Applies `mamba3_bwd_stage2_force_nontma.patch` plus optional overlay patches, then compares correctness and timings against baseline.

## Local Validation

- Patch dry-run:
  - `mamba3_bwd_stage2_force_nontma.patch` applies to local `/home/dave/state-spaces-mamba/.../mamba3_mimo_bwd.py`.
  - all three pressure overlay patches dry-run cleanly after stage2.
  - direct-bias + rotary-onetmp overlay stack also dry-runs cleanly.
- Python compile:
  - `python -m py_compile scripts/modal_mamba3_bwd_bwd_pressure_reduce_benchmark.py scripts/modal_mamba3_stage2_force_nontma_benchmark.py upstream_prs/examples/13_tilelang_floormod_dbz/reproducer.py`
  - passed.

## Modal H200 Smoke

Run:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 15m \
  modal run scripts/modal_mamba3_bwd_bwd_pressure_reduce_benchmark.py \
  --shape-csv smoke \
  --variant-csv baseline,stage2_force_nontma,pressure_direct_bias,pressure_rotary_onetmp,pressure_dgamma_serial \
  --iters 3 --warmup 1
```

Artifacts: Modal volume `cppmega-mamba3-benchmarks`, `/mamba3_bwd_bwd_pressure_reduce_benchmark/20260429_175251`.

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | correctness vs baseline |
|---|---:|---:|---:|---|
| baseline | 0.0801 | 0.1634 | 0.2277 | reference |
| stage2_force_nontma | 0.0840 | 0.1636 | 0.2299 | zero diff |
| pressure_direct_bias | 0.0823 | 0.1736 | 0.2385 | zero diff |
| pressure_rotary_onetmp | 0.0818 | 0.1692 | 0.2345 | zero diff |
| pressure_dgamma_serial | 0.0819 | 0.1666 | 0.2332 | main grads zero diff; `DGAMMA_DIAG` max abs `2.78e-16` |

All variants compiled. `bwd_bwd` stayed non-WS and had zero TMA loads.

## Modal H200 Productionish

Run:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 15m \
  modal run scripts/modal_mamba3_bwd_bwd_pressure_reduce_benchmark.py \
  --shape-csv productionish \
  --variant-csv baseline,stage2_force_nontma,pressure_dgamma_serial \
  --iters 4 --warmup 1
```

Artifacts: Modal volume `cppmega-mamba3-benchmarks`, `/mamba3_bwd_bwd_pressure_reduce_benchmark/20260429_175921`.

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | speedup chain vs baseline | correctness vs baseline |
|---|---:|---:|---:|---:|---|
| baseline | 1.8738 | 3.7207 | 5.5584 | 1.0000x | reference |
| stage2_force_nontma | 1.8002 | 3.7047 | 5.4687 | 1.0164x | zero diff |
| pressure_dgamma_serial | 1.7989 | 3.8716 | 5.6335 | 0.9867x | main grads zero diff; `DGAMMA_DIAG` max abs `8.88e-16` |

## Recommendation

Discard these pressure variants for now.  They reduce obvious fragment pressure on paper, but smoke timings regress and the productionish `dgamma_serial` candidate makes `bwd_bwd` slower.  Keep `stage2_force_nontma` as the best current default and do not pursue another `bwd_bwd` WS/TMA num-stages sweep from this branch.
