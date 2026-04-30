# Mamba3 Mono CuTe Chunk Wave 7 - 2026-04-30

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Move the remaining tested tile-chain `state + apply` handoff and the
`DV`/`DMIMO_V` consumers into one scan-owner CuTe kernel.  Correctness remains
the priority; scalar BF16 copy mode stays in place.

## Files Changed

- `cppmega/megatron/cute_dsl_mimo/state_apply_consumers.py`
- `cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py`
- `scripts/modal_mamba3_mono_chunk_wave2.py`
- `docs/status/mamba3_mono_cute_chunk_wave7_2026_04_30.md`

## Prototype

Added `state_apply_consumers.py`, a fixed `64x64x64` BF16 CuTe tile:

- GEMM1 computes `state = K @ DStates`.
- GEMM2 computes and masks `LKQ = future_mask(K @ Q.T)`.
- Both `state` and masked `LKQ` are BF16 R2S-spilled only to swizzled shared
  memory.
- GEMM3 computes `apply = masked(LKQ) @ dPhi`.
- `apply` is BF16 R2S-spilled only to shared memory.
- Scalar in-kernel consumers read BF16-rounded `state` and `apply`, compute
  `dpsi = state.float() + apply.float()`, then write only FP32 `DV` and
  `DMIMO_V` outputs.

The Wave 5 scalar path and Wave 6 fused masked-apply path remain in the harness
for side-by-side correctness and timing.

## H200 Smoke

Command:

```bash
CPPMEGA_MODAL_APP_NAME=cppmega-mamba3-mono-chunk-wave7-lane-b-codex-h200 \
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

App: `ap-aq6JYUbEUesFRqr8FVSkoz`

Result:

- GPU: `NVIDIA H200`
- Correctness: pass
- Tolerance: `1e-5`
- Compile + first LKQ/fused-apply/fused-consumer launch: `2.771 s`
- Scalar correctness mode:
  - `structured_mod`: all checked diffs `0.0`
  - `random_seed`: `state=7.6294e-06`, `dpsi=7.6294e-06`,
    `dv=7.8604e-07`, `dmimo_v=4.3958e-07`, LKQ/apply diffs `0.0`
  - Timing: `105.939 us/chain` over `20` iters, includes torch mask handoff
- Wave 6 fused masked-apply mode:
  - `structured_mod`: all checked diffs `0.0`
  - `random_seed`: `state=7.6294e-06`, `apply=0.0`, `dpsi=7.6294e-06`,
    `dv=7.8604e-07`, `dmimo_v=4.3958e-07`
  - Timing: `71.360 us/chain` over `20` iters, no LKQ global output and no
    torch mask handoff
- Wave 7 fused state/apply consumer mode:
  - `structured_mod`: `dv=0.0`, `dmimo_v=0.0`
  - `random_seed`: `dv=7.8604e-07`, `dmimo_v=4.3958e-07`
  - Timing: `63.419 us/chain` over `20` iters, one launch and no
    LKQ/state/apply global outputs
- Estimated tile-only throughput:
  - scalar: `0.0148 TFLOP/s`
  - Wave 6 fused apply: `0.0220 TFLOP/s`
  - Wave 7 fused consumers: `0.0251 TFLOP/s`

Modal cleanup:

```bash
modal app stop -y ap-aq6JYUbEUesFRqr8FVSkoz
```

returned that the app was already stopped at `2026-04-30 11:44:20+00:00`.

## Materialization Status

For the tested Wave 7 fused-consumer path:

- `LKQ`: no global output; only BF16 swizzled shared memory.
- `state`: no global output; only BF16 swizzled shared memory.
- `apply`: no global output; only BF16 swizzled shared memory.
- `dpsi`: no global output; exists only as scalar float sums in the consumer
  loops.
- Global outputs that remain: final FP32 `DV` and `DMIMO_V`.
- Harness input-layout tensors that remain: pre-transposed `DStates.T` and
  `DPh.T`.

The older scalar path still materializes LKQ/masked-LKQ/state/apply globally.
The Wave 6 side-by-side path still materializes state/apply globally.

## Copy Strategy Status

Still scalar BF16 universal G2S copies for the 64x64 CuTe operands.  The new
consumer path uses direct scalar BF16 global loads for `V` and `mimo_v`, and
direct FP32 global stores for `DV` and `DMIMO_V`.  No 128-bit universal
G2S/S2G path was reintroduced.

## Next Blocker

Lift this one-chunk probe into the multi-chunk scan-owner bwd_bwd path:
thread the loop-carried state, include the diagonal/qk contributions, and
replace the scalar consumer loops with a layout-aware vectorized/warp-reduced
consumer once correctness survives the full TileLang parity boundary.
