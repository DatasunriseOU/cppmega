# Mamba3 Mono CuTe Chunk Wave 8 - 2026-04-30

Status: active
Canonical: none
Date: 2026-04-30
Scope: Bounded multi-chunk CuTe scan-owner prototype for the fused state/apply consumer path.

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Lift the Wave 7 one-chunk fused CuTe state/apply consumer probe into a small
multi-chunk reverse scan owner.  Correctness remains the priority; diagonal/qk
terms and production bwd_bwd integration stay deferred.

## Files Changed

- `cppmega/megatron/cute_dsl_mimo/state_apply_consumers.py`
- `cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py`
- `scripts/modal_mamba3_mono_chunk_wave2.py`
- `docs/status/mamba3_mono_cute_chunk_wave8_2026_04_30.md`

## Prototype

Added `MultiChunkStateApplyConsumersWGMMA`, a fixed `64x64` BF16 CuTe kernel
for `nchunks in {2, 4, 8}`:

- Runs one CTA as the scan owner for the chunk sequence.
- Processes chunks in reverse order.
- Keeps `state`, masked `LKQ`, `apply`, and `dpsi` off global memory.
- Carries `DStates.T` in FP32 registers as `carry_t`.
- BF16-spills `carry_t` to swizzled shared memory only when feeding the current
  chunk's `state = K @ DStates` GEMM.
- Updates the carried state after each chunk with `carry_t += dPhi.T @ Q`.
- Writes only final FP32 `DV` for all chunks and accumulated FP32 `DMIMO_V`.

The path is still a bounded prototype.  It keeps harness-side `Q.T` and
`DPh.T` tensors so the carry update can use K-major WGMMA operands.  This
matches the transpose workaround already needed in the wider P4 prototype.

## H200 Smoke

Command:

```bash
CPPMEGA_MODAL_APP_NAME=cppmega-mamba3-mono-chunk-wave8-lane-b-codex-20260430-h200-b \
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

App: `ap-tzdaTm79HGuhiUolb9cHO0`

Result:

- GPU: `NVIDIA H200`
- Overall correctness: pass
- Tolerance: `1e-5`
- Compile + first LKQ/fused-apply/fused-consumer launch: `2.629 s`
- Multi-chunk compile + correctness section: `4.111 s`
- One-chunk Wave 7 fused state/apply consumer:
  - `structured_mod`: `dv=0.0`, `dmimo_v=0.0`
  - `random_seed`: `dv=7.8604e-07`, `dmimo_v=4.3958e-07`
  - Timing: `75.445 us/chain` over `20` iters
- Multi-chunk fused scan:
  - `2` chunks:
    - `structured_mod`: `dv=0.0`, `dmimo_v=0.0`
    - `random_seed`: `dv=3.638e-12`, `dmimo_v=5.821e-11`
    - Timing: `76.171 us/scan`, `38.086 us/chunk`
  - `4` chunks:
    - `structured_mod`: `dv=0.0`, `dmimo_v=0.0`
    - `random_seed`: `dv=1.455e-11`, `dmimo_v=1.164e-10`
    - Timing: `78.264 us/scan`, `19.566 us/chunk`
  - `8` chunks:
    - `structured_mod`: `dv=0.0`, `dmimo_v=0.0`
    - `random_seed`: `dv=2.082e-06`, `dmimo_v=1.647e-06`
    - Timing: `77.262 us/scan`, `9.658 us/chunk`
- Baseline timings from the same run:
  - scalar CuTe tiles + torch mask: `131.182 us/chain`
  - Wave 6 fused masked apply: `78.728 us/chain`
  - Wave 7 fused consumers: `75.445 us/chain`

Modal cleanup:

```bash
modal app stop -y ap-tzdaTm79HGuhiUolb9cHO0
```

returned that the app was already stopped at `2026-04-30 12:02:52+00:00`.
The earlier failed tuning run app `ap-P7UsaFndiFDbFsm9Ggej06` was also already
stopped at `2026-04-30 12:01:29+00:00`.

## Materialization Status

For the tested Wave 8 multi-chunk path:

- `LKQ`: no global output; only BF16 swizzled shared memory per chunk.
- `state`: no global output; only BF16 swizzled shared memory per chunk.
- `apply`: no global output; only BF16 swizzled shared memory per chunk.
- `dpsi`: no global output; exists only as scalar float sums in consumer loops.
- Loop-carried state: FP32 registers across chunks; BF16 shared-memory spill
  only as the next chunk's state GEMM operand.
- Global outputs that remain: final FP32 `DV` and accumulated FP32 `DMIMO_V`.
- Harness input-layout tensors that remain: pre-transposed `Q.T` and `DPh.T`.

The older scalar path still materializes LKQ/masked-LKQ/state/apply globally.
The Wave 6 side-by-side path still materializes state/apply globally.  The Wave
7 one-chunk side-by-side path still uses `DStates.T`/`DPh.T` harness inputs.

## Copy Strategy Status

Still scalar BF16 universal G2S copies.  The multi-chunk path uses direct scalar
BF16 global loads for `V` and `mimo_v`, direct FP32 global stores for `DV` and
`DMIMO_V`, and no 128-bit universal G2S/S2G path.

## Next Blocker

Move from this bounded scan-owner probe to production `bwd_bwd`:

- Fold in `exp(dA_cs_rev)`, `exp(dA_cs)`, and chunk state scaling.
- Add diagonal and qk contributions before `DV`/`DMIMO_V`.
- Replace scalar consumer loops/global `DMIMO_V` read-modify-write with a
  layout-aware vectorized or warp-reduced consumer.
- Remove or internalize the harness transposes where CuTe operand layout allows
  it.
