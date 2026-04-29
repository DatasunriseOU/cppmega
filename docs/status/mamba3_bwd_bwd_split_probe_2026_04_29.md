# Mamba3 `bwd_bwd` Split Probe - 2026-04-29

Branch: `worker/mamba3-bwd-bwd-split-probe`

Base: `972608d` (`worker/mamba3-stage2-force-nontma`)

## Goal

Bounded structural probe for whether the `bwd_bwd` live set can be reduced by
splitting the chunk-local diagonal path away from the reverse-state path.

This is not a production rewrite. The patch appends a reduced-output TileLang
kernel to a temp copy of upstream `mamba3_mimo_bwd.py`:

`mamba_mimo_bwd_bwd_dgamma_diag_probe`

The probe computes only:

`DOUT/V/MIMO_V/MIMO_O/QK_DOT -> DGAMMA_DIAG`

It intentionally does not take `STATES`, reverse-carried state buffers, `Q`, or
`K`. That isolates the `dPhiO/PsiV/qk_dot/dqk_from_diag` diagonal subgraph from
the full `dstates` reverse loop.

## Files

- `scripts/modal_mamba3_bwd_bwd_split_probe.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_bwd_split_probe_dgamma.patch`
- `docs/status/mamba3_bwd_bwd_split_probe_2026_04_29.md`

## Local Checks

Command:

```bash
python -m py_compile scripts/modal_mamba3_bwd_bwd_split_probe.py
python scripts/modal_mamba3_bwd_bwd_split_probe.py --local-dry-run
```

Result:

- `py_compile`: pass
- patch dry-run/apply chain: pass
- patch chain:
  - `mamba3_bwd_stage2_force_nontma.patch`
  - `mamba3_bwd_bwd_split_probe_dgamma.patch`
- patched source markers:
  - `split_probe_defs=1`
  - `split_probe_kernel_defs=1`
  - `split_probe_uses_dstates=false`
  - `split_probe_uses_states=false`
  - `split_probe_uses_q_or_k=false`
  - `stage2_flat_qk_count=3`

## Modal H200 Smoke

Command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 15m \
  modal run scripts/modal_mamba3_bwd_bwd_split_probe.py \
  --shape-csv smoke --iters 5 --warmup 2
```

Run: `ap-c3SQlz8yGgrtw9LsJNvDyK`

Shape: `B=1,S=256,H=4,G=1,N=64,P=64,R=4,bf16`

Result:

- compile: pass (`bwd_fwd`, full `bwd_bwd`, `dgamma_probe`)
- correctness: `DGAMMA_DIAG` allclose, absmax diff `2.220446049250313e-16`
- chain mean: `0.2316 ms`
- full `bwd_bwd` mean: `0.1656 ms`
- standalone `dgamma_probe` mean: `0.0238 ms`
- artifact JSON: `/benchmarks/mamba3_bwd_bwd_split_probe_20260429_192509.json`

## Modal H200 Productionish

Command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 20m \
  modal run scripts/modal_mamba3_bwd_bwd_split_probe.py \
  --shape-csv productionish --iters 5 --warmup 2
```

Run: `ap-0K1ra5q3NNmYbkr9FkzGKn`

Shape: `B=4,S=4096,H=32,G=1,N=64,P=128,R=4,bf16`

Result:

- compile: pass (`bwd_fwd`, full `bwd_bwd`, `dgamma_probe`)
- correctness: `DGAMMA_DIAG` allclose, absmax diff `5.551115123125783e-16`
- chain mean: `5.5010 ms`
- full `bwd_bwd` mean: `3.7220 ms`
- standalone `dgamma_probe` mean: `0.3966 ms`
- `dgamma_probe / bwd_bwd`: `10.66%`
- artifact JSON: `/benchmarks/mamba3_bwd_bwd_split_probe_20260429_192141.json`

Compile source markers at tiny shape:

- `bwd_fwd`: `source_chars=41906`, `tma_load_count=4`, `tma_store_count=3`, `producer_guard=true`
- full `bwd_bwd`: `source_chars=89206`, `tma_load_count=0`, `producer_guard=false`
- `dgamma_probe`: `source_chars=9664`, `tma_load_count=0`, `producer_guard=false`

Both Modal apps completed and stopped after the local entrypoint exited.

## Recommendation

Discard a narrow `DGAMMA_DIAG`-only split as a production optimization.

The structural premise is valid: the chunk-local diagonal subgraph compiles as a
small independent kernel and matches the full kernel bit-exactly for the probed
output. However, on productionish H200 it costs about `0.40 ms` by itself,
roughly `10.7%` of full `bwd_bwd` and about `7.2%` of the chain. A real split would
need to remove at least that much from the remaining `bwd_bwd` kernel before it
breaks even, and this bounded probe does not demonstrate that.

Keep the probe as evidence that a larger split is technically plausible, but
only pursue a fuller state-path vs chunk-local rewrite if the next experiment
can also remove the diagonal work from the original `bwd_bwd` and measure a net
chain speedup. The current bounded result is a discard for immediate production.
