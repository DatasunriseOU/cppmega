# Mamba3 Mono CuTe Chunk Wave 5 - 2026-04-30

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Build on the Wave 4 hand-written CuTe `64x64x64` BF16 GEMM correctness fix and
start composing scan-owner chunk math with that tile as the oracle/building
block.  Correctness remains the priority; wide copies stay disabled until the
swizzled-smem copy path is layout-aware.

## Files Changed

- `cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py`
- `scripts/modal_mamba3_mono_chunk_wave2.py`
- `docs/status/mamba3_mono_cute_chunk_wave5_2026_04_30.md`

## Prototype

Added `lkq_tile_chain_test.py`, a bounded CuTe DSL composition probe for one
chunk owner:

- Fixed shape: `chunk=16`, `rank=4`, `fcs=64`, `N=64`, `P=64`.
- Reuses the correct Wave 4 `SingleGemmWGMMA` tile with scalar BF16
  G2S/S2G copies.
- Runs three CuTe tile invocations:
  - `state = K @ DStates`
  - `lkq = K @ Q^T`
  - `apply = future_mask(lkq) @ dPhi`
- Composes `dpsi = state + apply`.
- Checks scalar `DV` and `DMIMO_V` consumers from `dpsi`.

This is not yet one fused CuTe kernel.  The mask is applied by a torch op
between CuTe tile calls, with explicit synchronization around the torch/CuTe
handoff in timing mode.  That makes the prototype conservative but gives a
validated LKQ/state composition boundary before moving the mask and consumers
inside one scan-owner kernel.

## Modal Harness

Added mode:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

App: `ap-vvUCExbahebW1ndv0Ca4JG`

Result:

- GPU: `NVIDIA H200`
- Correctness: pass
- Tolerance: `1e-5`
- Compile + first LKQ launch: `0.961 s`
- Copy strategy: `scalar_bf16_universal_g2s_s2g`
- Cases:
  - `structured_mod`: all checked diffs `0.0`
  - `random_seed`:
    - `state`: `7.6294e-06`
    - `lkq`: `0.0`
    - `masked_lkq`: `0.0`
    - `apply`: `0.0`
    - `dpsi`: `7.6294e-06`
    - `dv`: `7.8604e-07`
    - `dmimo_v`: `4.3958e-07`
- Timing: `101.974 us/chain` over `20` iterations, including three scalar-copy
  CuTe tile launches plus the torch mask handoff.
- Estimated tile-only math: `1,572,864` flops/chain.
- Estimated tile throughput: `0.0154 TFLOP/s`.

The Wave 4 single-tile oracle still passes on the same edited harness:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py --mode cute-gemm
```

Result:

- Correctness: pass for `identity_transpose`, `structured_mod`,
  `random_seed_42`.
- Timing: `28.0227 us/iter`.

## Local Smoke

Syntax:

```bash
python -m py_compile \
  cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  tools/probes/mamba3_mono_chunk_smoke.py \
  cppmega/megatron/mamba3_mono_chunk_skeleton.py
```

CUDA WMMA fallback smoke:

```bash
python tools/probes/mamba3_mono_chunk_smoke.py \
  --B 1 --S 16 --H 1 --P 64 --atol 2e-2 --bench-iters 5 --bench-warmup 1
```

Result:

- Correctness: pass.
- Diffs:
  - `dv`: `2.7545e-04`
  - `dmimo_v`: `3.5092e-04`
  - `dk_diag`: `2.4212e-10`
  - `dq_diag`: `2.8837e-10`
  - `lkq_checksum`: `2.9802e-06`
- Timing: `144.365 us/iter` for the tiny local smoke shape.

## Copy Strategy Status

Still scalar BF16 universal copies for the hand-written CuTe tile.  No 128-bit
universal G2S/S2G copies were reintroduced.  The failed first timed chain run
showed the torch mask handoff needs explicit ordering when mixed with CuTe DSL
launches; the current timing path adds synchronization around that temporary
handoff.

Next copy step remains unchanged: reintroduce wider copies only through a
layout-aware path for swizzled smem, preferably quack-style TMA/cp.async or a
validated smem-layout copy utility.

## Next Blocker

The chain is still three CuTe tile launches plus a torch mask, not one
scan-owner CuTe kernel.  The next blocker is moving `future_mask(lkq) @ dPhi`
into the CuTe kernel as a masked epilogue / second WGMMA consumer while keeping
the pointer-swizzled smem layout correct, then adding the remaining DK/DQ
families without global LKQ materialization.
