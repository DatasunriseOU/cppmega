# Mamba3 Mono CuTe Chunk Wave 9 - 2026-04-30

Status: active
Canonical: none
Date: 2026-04-30
Scope: Add same-time qk diagonal semantics to the Wave8 fused multi-chunk CuTe scan-owner probe.

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Add the next production semantic to the tested Wave8 fused CuTe path without
changing its bounded ownership or BF16 rounding contract.  Correctness remains
the priority; dA scaling and full production integration stay deferred.

## Files Changed

- `cppmega/megatron/cute_dsl_mimo/state_apply_consumers.py`
- `cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py`
- `scripts/modal_mamba3_mono_chunk_wave2.py`
- `docs/status/mamba3_mono_cute_chunk_wave9_2026_04_30.md`

## Prototype

Extended `MultiChunkStateApplyConsumersWGMMA` with two new multi-chunk inputs:

- `qk_dot`: BF16, shaped `[nchunks, chunk, R, R]`
- `gamma`: FP32, shaped `[nchunks, chunk]`

The fused consumer now adds the same-time qk diagonal term before both `DV` and
`DMIMO_V` reductions:

```text
dpsi[t, r_in, p] += gamma[t] *
    sum_r_out qk_dot[t, r_out, r_in] * dPhi[t, r_out, p]
```

This matches the production `qk_dot.transpose(-1, -2) @ dPhi` contribution for
the same-time block, but only inside the Wave8 bounded probe contract:

- state is still `BF16(K @ BF16(carry).T)`;
- masked LKQ and apply are still BF16-spilled through shared memory;
- `dpsi` remains a scalar FP32 sum consumed in-kernel;
- the production combined `dPsiV_D.to(bf16)` boundary is not introduced yet.

The torch reference and deterministic/random multi-chunk cases now include
`qk_dot` and `gamma`, and compare final `DV`/`DMIMO_V` for `nchunks in {2,4,8}`.

## H200 Smoke

Command:

```bash
CPPMEGA_MODAL_APP_NAME=cppmega-mamba3-mono-chunk-wave9-lane-b-codex-20260430-h200-b \
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

App: `ap-cmQuEeFzrjpDERXHe71fPG`

Result:

- GPU: `NVIDIA H200`
- Overall correctness: pass
- Tolerance: `1e-5`
- Compile + first LKQ/fused-apply/fused-consumer launch: `2.444 s`
- Multi-chunk compile + correctness section: `7.488 s`
- One-chunk Wave7 fused state/apply consumer:
  - `structured_mod`: `dv=0.0`, `dmimo_v=0.0`
  - `random_seed`: `dv=7.8604e-07`, `dmimo_v=4.3958e-07`
  - Timing: `62.142 us/chain` over `20` iters
- Multi-chunk fused scan with qk diagonal addend:
  - `2` chunks:
    - `structured_mod`: `dv=1.490e-07`, `dmimo_v=4.768e-07`
    - `random_seed`: `dv=5.821e-11`, `dmimo_v=1.746e-10`
    - Timing: `76.632 us/scan`, `38.316 us/chunk`
  - `4` chunks:
    - `structured_mod`: `dv=3.576e-07`, `dmimo_v=9.537e-07`
    - `random_seed`: `dv=1.164e-10`, `dmimo_v=2.910e-10`
    - Timing: `82.056 us/scan`, `20.514 us/chunk`
  - `8` chunks:
    - `structured_mod`: `dv=4.768e-07`, `dmimo_v=1.669e-06`
    - `random_seed`: `dv=1.174e-06`, `dmimo_v=1.922e-06`
    - Timing: `79.344 us/scan`, `9.918 us/chunk`
- Baseline timings from the same run:
  - scalar CuTe tiles + torch mask: `108.042 us/chain`
  - Wave6 fused masked apply: `63.229 us/chain`
  - Wave7 fused consumers: `62.142 us/chain`

Modal cleanup:

```bash
modal app stop -y ap-cmQuEeFzrjpDERXHe71fPG
```

returned that the app was already stopped at `2026-04-30 12:23:13+00:00`.

## Materialization Status

For the tested Wave9 multi-chunk path:

- `LKQ`: no global output; only BF16 swizzled shared memory per chunk.
- `state`: no global output; only BF16 swizzled shared memory per chunk.
- `apply`: no global output; only BF16 swizzled shared memory per chunk.
- `dpsi`: no global output; scalar FP32 sums include state, apply, and qk
  diagonal addend inside the consumer loops.
- Loop-carried state: FP32 registers across chunks; BF16 shared-memory spill
  only as the next chunk's state GEMM operand.
- New global inputs: BF16 `qk_dot` and FP32 `gamma`.
- Global outputs that remain: final FP32 `DV` and accumulated FP32 `DMIMO_V`.
- Harness input-layout tensors that remain: pre-transposed `Q.T` and `DPh.T`.

The older scalar path still materializes LKQ/masked-LKQ/state/apply globally.
The Wave6 side-by-side path still materializes state/apply globally.  The Wave7
one-chunk side-by-side path still uses `DStates.T`/`DPh.T` harness inputs and
does not include the qk addend.

## Semantics Covered

Now covered for the tested fused multi-chunk probe:

- reverse scan-owner loop over `2/4/8` chunks;
- off-global state/LKQ/apply/dpsi;
- FP32 loop-carried `carry_t += dPhi.T @ Q`;
- final `DV` and accumulated `DMIMO_V`;
- same-time qk diagonal `qk_dot * gamma * dPhi` contribution to `dpsi`.

Still missing for production `bwd_bwd`:

- `exp(dA_cs_rev)` state scaling;
- `exp(segsum)` intrachunk LKQ scaling;
- `dstates *= exp(dA_cs_last)` and `dPhi * exp(dA_cs)` carry update scaling;
- optional `D * dPhi` same-time diagonal addend;
- production combined `dPsiV_D.to(bf16)` before `DV`/`DMIMO_V`;
- full diagonal `DGAMMA_DIAG`, diagonal `DK/DQ`, and non-diagonal `DK/DQ`;
- scalar consumer replacement with vectorized/warp-reduced consumers;
- global `DMIMO_V` ownership beyond the one-CTA bounded probe;
- internalizing `Q.T` and `DPh.T` harness transposes.

## Wave10 Task

Add dA scaling to the same multi-chunk fused path:

1. Thread FP32 `dA_cs`, `dA_cs_rev`, and `segsum` into the Wave9 kernel.
2. Scale the state term by `exp(dA_cs_rev[t])`.
3. Scale reverse-causal LKQ by `exp(segsum[col_t, row_t])`.
4. Update the carry as
   `carry_t = exp(dA_cs_last) * carry_t + (dPhi * exp(dA_cs)).T @ Q`.
5. Keep the existing qk addend active and compare `DV`/`DMIMO_V` against the
   torch reference for `2/4/8` chunks.
