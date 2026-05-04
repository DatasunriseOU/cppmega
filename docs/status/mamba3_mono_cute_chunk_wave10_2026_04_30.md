# Mamba3 Mono CuTe Chunk Wave 10 - 2026-04-30

Status: active
Canonical: none
Date: 2026-04-30
Scope: Add dA, segsum, and scaled carry semantics to the Wave9 fused multi-chunk CuTe scan-owner probe.

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Add the next production semantics to the tested Wave9 fused CuTe path while
keeping the mini probe bounded to `nchunks in {2, 4, 8}` and prioritizing
correctness over production ownership and vectorization.

## Files Changed

- `cppmega/megatron/cute_dsl_mimo/state_apply_consumers.py`
- `cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py`
- `scripts/modal_mamba3_mono_chunk_wave2.py`
- `docs/status/mamba3_mono_cute_chunk_wave10_2026_04_30.md`

## Prototype

Extended `MultiChunkStateApplyConsumersWGMMA` with three FP32 multi-chunk
inputs:

- `dA_cs`: `[nchunks, chunk]`
- `dA_cs_rev`: `[nchunks, chunk]`
- `segsum`: `[nchunks, chunk, chunk]`

The fused multi-chunk consumer now covers:

- `state = BF16((K @ BF16(carry).T) * exp(dA_cs_rev[t]))`;
- `masked_LKQ = BF16((K @ Q.T) * exp(segsum[col_t, row_t]))` for
  `row_t < col_t`;
- `carry_t = exp(dA_cs_last) * carry_t + BF16(dPhi * exp(dA_cs)).T @ Q`;
- existing Wave9 qk diagonal addend:
  `dpsi[t,r,p] += gamma[t] * sum_o qk_dot[t,o,r] * dPhi[t,o,p]`.

This still follows the probe's BF16 shared-memory operand contract.  The
production combined `dPsiV_D.to(bf16)` boundary is still not introduced.

## H100 Mini Smoke

Command:

```bash
CPPMEGA_MODAL_APP_NAME=cppmega-mamba3-mono-chunk-wave10-lane-b-codex-20260430-h100-b \
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

App: `ap-eybkdGkU4KlD6ZO6yIc9gF`

Result:

- GPU: `NVIDIA H100 80GB HBM3`
- GPU spec: `H100`
- Overall correctness: pass
- Tolerance: `1e-5`
- Compile + first LKQ/fused-apply/fused-consumer launch: `1.701 s`
- Multi-chunk compile + correctness section: `7.357 s`
- One-chunk Wave7 fused state/apply consumer:
  - `structured_mod`: `dv=0.0`, `dmimo_v=0.0`
  - `random_seed`: `dv=7.8604e-07`, `dmimo_v=4.3958e-07`
  - Timing: `64.386 us/chain` over `20` iters
- Multi-chunk fused scan with qk + dA/segsum/carry scaling:
  - `2` chunks:
    - `structured_mod`: `dv=1.490e-07`, `dmimo_v=4.768e-07`
    - `random_seed`: `dv=2.685e-09`, `dmimo_v=7.276e-10`
    - Timing: `98.131 us/scan`, `49.066 us/chunk`
  - `4` chunks:
    - `structured_mod`: `dv=3.576e-07`, `dmimo_v=9.537e-07`
    - `random_seed`: `dv=2.369e-06`, `dmimo_v=9.565e-07`
    - Timing: `104.536 us/scan`, `26.134 us/chunk`
  - `8` chunks:
    - `structured_mod`: `dv=4.768e-07`, `dmimo_v=2.146e-06`
    - `random_seed`: `dv=6.041e-07`, `dmimo_v=2.240e-07`
    - Timing: `108.845 us/scan`, `13.606 us/chunk`
- Baseline timings from the same run:
  - scalar CuTe tiles + torch mask: `126.195 us/chain`
  - Wave6 fused masked apply: `69.090 us/chain`
  - Wave7 fused consumers: `64.386 us/chain`

Modal cleanup:

```bash
modal app stop -y ap-eybkdGkU4KlD6ZO6yIc9gF
modal app stop -y ap-HZ8gWe3Kd3Amh4mfTk31n8
modal app list
```

Both Wave10 apps were already stopped; `modal app list` showed stopped apps with
`0` tasks.

## Materialization Status

For the tested Wave10 multi-chunk path:

- `LKQ`: no global output; BF16 swizzled shared memory after segsum scaling.
- `state`: no global output; BF16 swizzled shared memory after
  `dA_cs_rev` scaling.
- `apply`: no global output; BF16 swizzled shared memory.
- `dpsi`: no global output; scalar FP32 sums include state, apply, qk diagonal.
- Loop-carried state: FP32 registers across chunks; BF16 shared-memory spill
  only as the next chunk's state GEMM operand.
- New global inputs: FP32 `dA_cs`, `dA_cs_rev`, and `segsum`.
- Global outputs that remain: final FP32 `DV` and accumulated FP32 `DMIMO_V`.
- Harness input-layout tensors that remain: pre-transposed `Q.T` and `DPh.T`.

## Semantics Covered

Now covered for the tested fused multi-chunk probe:

- reverse scan-owner loop over `2/4/8` chunks;
- off-global state/LKQ/apply/dpsi;
- same-time qk diagonal `qk_dot * gamma * dPhi` contribution to `dpsi`;
- `exp(dA_cs_rev)` state scaling;
- `exp(segsum[col_t,row_t])` intrachunk LKQ scaling;
- `exp(dA_cs_last)` carry decay;
- `(dPhi * exp(dA_cs)).T @ Q` carry update scaling.

Still missing for production `bwd_bwd`:

- optional `D * dPhi` same-time diagonal addend;
- production combined `dPsiV_D.to(bf16)` before `DV`/`DMIMO_V`;
- full diagonal `DGAMMA_DIAG`, diagonal `DK/DQ`, and non-diagonal `DK/DQ`;
- scalar consumer replacement with vectorized/warp-reduced consumers;
- global `DMIMO_V` ownership beyond the one-CTA bounded probe;
- internalizing `Q.T` and `DPh.T` harness transposes;
- production NAM56R integration and ownership in the real Mamba3 backward path.

## R&D Path Status

This remains the best R&D path.  The mini fused path now has the core scan and
decay semantics needed before productionizing, and it still avoids global
state/LKQ/apply/dpsi materialization.  The next useful step is not another
semantic addend unless `D * dPhi` is needed for parity; it is vectorizing the
consumers and resolving production `DMIMO_V` ownership.
