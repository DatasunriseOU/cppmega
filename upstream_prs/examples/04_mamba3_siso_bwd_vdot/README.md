# Reproducer: PR #04 - eliminate redundant `V @ dO^T` in Mamba-3 SISO bwd

Template: [`upstream_prs/04_mamba3_siso_bwd_eliminate_redundant_vdot.md`](../../04_mamba3_siso_bwd_eliminate_redundant_vdot.md)
Target:   [`state-spaces/mamba` -> `mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py)

## What it does

1. Imports the installed `mamba_ssm` (the stock, unpatched kernel).
2. Re-execs the source of `mamba3_siso_bwd.py` with two text-level substitutions
   that collapse the three redundant `tl.dot(V, dO^T)` computations into a single
   `vdot_masked` value reused by dADT, dK, and dQ.  The result is a fresh
   `@triton.jit` kernel object.
3. Swaps the kernel that `compute_dqkv` dispatches to, runs
   `mamba3_siso_combined` forward + backward on identical RNG state, and
   compares every returned gradient tensor with the stock kernel.
4. Times both variants with CUDA events (warmup + `--iters` medians, subtracting
   a forward-only baseline to isolate the backward).

Exits non-zero if any gradient exceeds `--rel-tol` or `--abs-tol`.

## Run

```bash
# From a machine with CUDA and the mamba_ssm (state-spaces) fork installed,
# e.g. h200_1:
source /mnt/data/venv/bin/activate
cd /path/to/cppmega/upstream_prs/examples/04_mamba3_siso_bwd_vdot
python reproducer.py
```

Useful flags:

- `--seqlen 2048 --nheads 32 --headdim_qk 128 --headdim_v 128` (medium config)
- `--seqlen 256 --batch 2 --nheads 8 --headdim_qk 64 --headdim_v 64` (tiny, for quick sanity)
- `--iters 50 --warmup 10` (tighter timing)
- `--rel-tol 1e-3 --abs-tol 1e-4` (template claims bitwise-identical; we default
  to 2e-3 to absorb non-deterministic autotune reorderings)

## Expected output

On H200 + Triton 3.7 nightly the template reports that Triton's CSE already
fuses the three dots, so correctness is the main signal:

- `CORRECTNESS OK` for all of `Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z`
- `bwd delta` near zero (single-digit percent or noise) on recent Triton
- On older Triton (<3.4) or non-H100/H200 arches the patched kernel can reach
  the full ~2-redundant-GEMMs savings

## Notes / limitations

- The reproducer does NOT modify the installed `mamba_ssm` package.  It
  re-execs the bwd module source in a private namespace and monkey-patches the
  kernel reference for the duration of the script.
- The forward pass is untouched by this PR, so `fwd max |orig - patched|`
  should be exactly `0.0`.
- Both the stock and patched kernels go through `@triton.autotune`; we include
  `--warmup 5` to let the first-run search complete before timing.
