# Mamba3 Mono CuTe Chunk Wave30 Lane B - 2026-04-30

Status: R&D only
Date: 2026-04-30
Scope: Close the production `dPsiV_D.to(bf16)` semantic boundary in the
bounded fused CuTe multi-chunk scan-owner path.

## Change

- Updated `MultiChunkStateApplyConsumersWGMMA` so each scalar consumer now
  forms `state + apply + D*dPhi + qk_diag`, converts that combined `dpsi` value
  to BF16, then widens to FP32 for the `DV` and `DMIMO_V` accumulations.
- Updated the PyTorch multi-chunk harness reference to use the same combined
  BF16 boundary before `_scalar_consumers`.
- Kept `dPsiV_D` non-materialized globally; the boundary is local to the fused
  consumer loops.
- Added source-level Wave30 contract tests covering the boundary placement and
  harness metadata.

## H100 Validation

Command:

```bash
CPPMEGA_MODAL_APP_NAME=cppmega-mamba3-mono-chunk-wave30-lane-b-dpsiv-bf16-boundary-h100 \
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

App: `ap-5EmgYZrW6IYgk1jJj56Mdt`

Result:

- GPU: `NVIDIA H100 80GB HBM3`
- Overall correctness: pass
- Tolerance: `1e-5`
- Compile + first LKQ/fused-apply/fused-consumer launch: `1.682 s`
- Multi-chunk compile + correctness section: `7.853 s`
- Peak CUDA memory: `32.89 MiB` allocated, `34.00 MiB` reserved

Timings over 20 iterations:

| Mode | Time |
| --- | ---: |
| Scalar CuTe tiles + torch mask | 159.674 us/chain |
| Wave6 fused masked apply | 80.235 us/chain |
| Wave7 fused state/apply consumers | 73.582 us/chain |
| Wave30 multi-chunk fused, 2 chunks | 117.912 us/scan, 58.956 us/chunk |
| Wave30 multi-chunk fused, 4 chunks | 119.040 us/scan, 29.760 us/chunk |
| Wave30 multi-chunk fused, 8 chunks | 117.334 us/scan, 14.667 us/chunk |

Correctness maxima for the Wave30 multi-chunk fused path:

| Case | DV max abs | DMIMO_V max abs |
| --- | ---: | ---: |
| structured, 2 chunks | 0.000e+00 | 0.000e+00 |
| random, 2 chunks | 0.000e+00 | 5.821e-11 |
| structured, 4 chunks | 0.000e+00 | 0.000e+00 |
| random, 4 chunks | 2.121e-06 | 1.859e-06 |
| structured, 8 chunks | 0.000e+00 | 2.384e-07 |
| random, 8 chunks | 6.333e-07 | 2.645e-07 |

## Local Checks

```bash
python -m py_compile \
  cppmega/megatron/cute_dsl_mimo/state_apply_consumers.py \
  cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py \
  tests/test_mamba3_cute_wave30_dpsiv_bf16_boundary_contract.py

pytest -q \
  tests/test_mamba3_cute_wave28_copy_contract.py \
  tests/test_mamba3_cute_wave29_d_direct_contract.py \
  tests/test_mamba3_cute_wave30_dpsiv_bf16_boundary_contract.py
```

Result: `7 passed`.

Local GPU note: `nvidia-smi` reports `NVIDIA GB10`; no local GPU kernel run was
used for final validation.  The GPU execution above used H100 only.  H200 was
not used.

## Production Status

Not safe for production main.  This wave closes the `dPsiV_D.to(bf16)`
boundary for the bounded one-head CuTe prototype, but it is still R&D-only:
full `DGAMMA_DIAG`/`DK`/`DQ`, vectorized or warp-reduced consumers, production
`DMIMO_V` ownership across CTAs, internal `Q.T`/`DPh.T`, and NAM56R integration
remain open.

Timing is worse than Wave29 in the same 20-iteration harness
(`8 chunks`: `117.334 us/scan` here vs `107.701 us/scan` in Wave29), but the
semantic coverage is closer to production because the combined `dPsiV_D` value
is now rounded to BF16 before both consumers.

## Modal Cleanup

Apps started in this lane:

- `ap-5EmgYZrW6IYgk1jJj56Mdt` - H100 Wave30 validation; completed and stopped
  when the local entrypoint exited.

Final `modal app list` showed `ap-5EmgYZrW6IYgk1jJj56Mdt` stopped with
`0` tasks.  Two unrelated `cppmega-wave...` ephemeral apps were still running
and were left alone.
