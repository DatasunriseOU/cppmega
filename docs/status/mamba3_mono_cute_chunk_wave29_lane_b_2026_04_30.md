# Mamba3 Mono CuTe Chunk Wave29 Lane B - 2026-04-30

Status: R&D only
Date: 2026-04-30
Scope: Add the direct `D * dPhi` same-time diagonal contribution to the
bounded fused CuTe multi-chunk scan-owner path.

## Change

- Threaded a one-element FP32 `D` tensor through
  `MultiChunkStateApplyConsumersWGMMA`.
- Added `dpsi[f,p] += D * dPhi[f,p]` inside both fused consumer loops:
  - `DV[t,p] = sum_r dpsi[t,r,p] * MIMO_V[r,p]`
  - `DMIMO_V[r,p] += sum_t dpsi[t,r,p] * V[t,p]`
- Updated the multi-chunk reference and structured/random H100 cases so `D` is
  nonzero and correctness covers the direct contribution.
- Added peak CUDA memory reporting to the LKQ chain harness.
- Left the Wave28 uint4 G2S path opt-in only; default remains scalar BF16 G2S.

## H100 Validation

Command:

```bash
CPPMEGA_MODAL_APP_NAME=cppmega-mamba3-mono-chunk-wave29-lane-b-d-direct-memory-h100 \
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

App: `ap-gtGB31s5auw4HvvyjfOCuk`

Result:

- GPU: `NVIDIA H100 80GB HBM3`
- Overall correctness: pass
- Tolerance: `1e-5`
- Compile + first LKQ/fused-apply/fused-consumer launch: `1.487 s`
- Multi-chunk compile + correctness section: `6.766 s`
- Peak CUDA memory: `32.89 MiB` allocated, `34.00 MiB` reserved

Timings over 20 iterations:

| Mode | Time |
| --- | ---: |
| Scalar CuTe tiles + torch mask | 105.930 us/chain |
| Wave6 fused masked apply | 64.730 us/chain |
| Wave7 fused state/apply consumers | 63.917 us/chain |
| Wave29 multi-chunk fused, 2 chunks | 104.470 us/scan, 52.235 us/chunk |
| Wave29 multi-chunk fused, 4 chunks | 105.242 us/scan, 26.310 us/chunk |
| Wave29 multi-chunk fused, 8 chunks | 107.701 us/scan, 13.463 us/chunk |

Correctness maxima for the Wave29 multi-chunk fused path:

| Case | DV max abs | DMIMO_V max abs |
| --- | ---: | ---: |
| structured, 2 chunks | 1.490e-07 | 4.768e-07 |
| random, 2 chunks | 7.312e-10 | 2.328e-09 |
| structured, 4 chunks | 3.576e-07 | 9.537e-07 |
| random, 4 chunks | 2.121e-06 | 2.036e-06 |
| structured, 8 chunks | 4.768e-07 | 1.550e-06 |
| random, 8 chunks | 6.333e-07 | 2.752e-07 |

Local source checks:

```bash
python -m py_compile \
  cppmega/megatron/cute_dsl_mimo/state_apply_consumers.py \
  cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py

pytest -q \
  tests/test_mamba3_cute_wave28_copy_contract.py \
  tests/test_mamba3_cute_wave29_d_direct_contract.py
```

Result: `5 passed`.

## Materialization Status

For the tested Wave29 multi-chunk path:

- `LKQ`: no global output; BF16 swizzled shared memory after segsum scaling.
- `state`: no global output; BF16 swizzled shared memory after
  `dA_cs_rev` scaling.
- `apply`: no global output; BF16 swizzled shared memory.
- `dpsi`: no global output; scalar FP32 sums now include state, apply, direct
  `D * dPhi`, and qk diagonal.
- Loop-carried state: FP32 registers across chunks; BF16 shared-memory spill
  only as the next chunk's state GEMM operand.
- Global outputs that remain: final FP32 `DV` and accumulated FP32 `DMIMO_V`.
- Harness input-layout tensors that remain: pre-transposed `Q.T` and `DPh.T`.

## Production Status

Not safe for production main.  This is a bounded one-head mini CuTe prototype
that now covers the direct `D * dPhi` addend, but still lacks the production
combined `dPsiV_D.to(bf16)` boundary, full `DGAMMA_DIAG`/`DK`/`DQ`, vectorized
or warp-reduced consumers, production `DMIMO_V` ownership across CTAs, internal
`Q.T`/`DPh.T`, and NAM56R integration.

## Modal Cleanup

Apps started in this lane:

- `ap-6fclieiAICxsXr8hknFtpf` - first PTY attempt, interrupted before probe
  body; stopped with 0 tasks.
- `ap-C8KB6Pi6ZNQEaCm0C5FmR1` - retry without memory reporting; pass.
- `ap-gtGB31s5auw4HvvyjfOCuk` - final H100 validation; pass.

All Wave29 `cppmega-mamba3...` apps completed or were already stopped through
the local entrypoint.  Final `modal app list` showed those apps stopped with
`0` tasks; two unrelated `cppmega-wave...` ephemeral apps were still running
and were left alone.
