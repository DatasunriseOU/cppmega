# Mamba3 Mono CuTe Chunk Wave 6 - 2026-04-30

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Move `future_mask(lkq) @ dPhi` out of the Python/Torch handoff and into a
single CuTe kernel path so LKQ does not round-trip through global memory for the
tested fused path.  Keep the Wave 5 scalar-copy path as the correctness mode.

## Files Changed

- `cppmega/megatron/cute_dsl_mimo/masked_lkq_apply.py`
- `cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py`
- `scripts/modal_mamba3_mono_chunk_wave2.py`
- `docs/status/mamba3_mono_cute_chunk_wave6_2026_04_30.md`

## Prototype

Added `masked_lkq_apply.py`, a fixed `64x64x64` BF16 CuTe tile:

- GEMM1 computes `lkq = K @ Q.T`.
- A per-accumulator future mask zeros entries whose chunk time is not future.
- Masked LKQ is converted to BF16 and written only to swizzled shared memory
  through the existing R2S path.
- GEMM2 consumes that shared LKQ directly for
  `apply = future_mask(lkq) @ dPhi`.
- The kernel writes only the BF16 `apply` tile to global memory.

The full scan-owner fusion is still not complete.  The Wave 6 fused path keeps:

- `state = K @ DStates` as the scalar-copy CuTe GEMM output.
- `apply` as a fused-kernel global output.
- `dpsi`, `DV`, and `DMIMO_V` as torch-side scalar correctness consumers.

## H200 Smoke

Command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

App: `ap-OTeVyEbeKiPQYGelwEWYgD`

Result:

- GPU: `NVIDIA H200`
- Correctness: pass
- Tolerance: `1e-5`
- Compile + first LKQ/fused apply launch: `1.274 s`
- Scalar correctness mode:
  - `structured_mod`: all checked diffs `0.0`
  - `random_seed`: `state=7.6294e-06`, `dpsi=7.6294e-06`,
    `dv=7.8604e-07`, `dmimo_v=4.3958e-07`, LKQ/apply diffs `0.0`
  - Timing: `104.710 us/chain` over `20` iters, includes torch mask handoff
- Fused masked-apply mode:
  - `structured_mod`: all checked diffs `0.0`
  - `random_seed`: `state=7.6294e-06`, `apply=0.0`, `dpsi=7.6294e-06`,
    `dv=7.8604e-07`, `dmimo_v=4.3958e-07`
  - Timing: `63.550 us/chain` over `20` iters, no LKQ global output and no
    torch mask handoff
- Estimated tile-only throughput:
  - scalar: `0.0150 TFLOP/s`
  - fused: `0.0247 TFLOP/s`

Modal stop status:

```bash
modal app stop ap-OTeVyEbeKiPQYGelwEWYgD
```

returned that the app was already stopped at `2026-04-30 11:28:34+00:00`.

## LKQ Materialization Status

Eliminated for the tested fused masked-apply path.  The fused kernel signature
has no LKQ output, and the only LKQ materialization is the BF16 R2S spill into
swizzled shared memory that immediately feeds the second WGMMA.  The scalar
correctness path and torch reference still materialize LKQ.

## Copy Strategy Status

Still scalar BF16 universal G2S/S2G copies for both the scalar GEMM oracle and
the fused masked-apply tile.  No 128-bit universal copies were reintroduced.

## Next Blocker

Fuse `state + apply` and the `DV` / `DMIMO_V` scalar consumers into the same
scan-owner kernel, then remove the remaining global `state` and `apply` tiles.
After that, reintroduce wider copies only through a layout-aware smem path.
