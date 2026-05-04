# Mamba3 CUDA Full bwd_bwd AB Harness Wave 8 - 2026-04-30

Branch: `worker/mamba3-cuda-full-bwd-ab`

Base: wave7 CUDA warp-owner commit `d399cf2`

Lane C goal: build the integration and AB harness around the full custom CUDA
`bwd_bwd` candidate. Lane A/B own kernel experiments; this lane owns packaging,
correctness, timing, and readiness criteria.

## Implemented

Added:

- `scripts/modal_mamba3_cuda_full_bwd_ab.py`

The harness runs in one Modal app and compares:

1. TileLang upstream `baseline`.
2. TileLang stage2 `(bf_num_stages=1, bb_num_stages=0)` as `stage2_bf1_bb0`.
3. Branch CUDA prototype components from wave7:
   - wave6 chunk-warp diagonal slice;
   - wave7 `qk_dot -> dPsiV -> dV` slice;
   - wave7 combined diagonal plus qk/dV one-launch slice.

The run uses CUDA events, TileLang source metadata, CUDA kernel metadata, and
`torch.cuda` peak memory counters only. It does not rely on NCU.

The script also has cheap alternate paths:

```text
CPPMEGA_MODAL_GPU=H100:2 modal run scripts/modal_mamba3_cuda_full_bwd_ab.py ...
CPPMEGA_MODAL_GPU=B200:1 modal run scripts/modal_mamba3_cuda_full_bwd_ab.py ...
```

B200 was not run in this wave because it is optional and should not block on
capacity. The CUDA extension arch is taken from the active device when possible.

## Validation

Local:

```text
python -m py_compile scripts/modal_mamba3_cuda_full_bwd_ab.py
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave7_chunk_owner_cuda.py \
  scripts/modal_mamba3_stage2_force_nontma_benchmark.py
git diff --check
```

Modal H200:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 1800s \
  modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
  --run-id wave8_h200_ab_20260430_2 \
  --shape-csv smoke,productionish \
  --iters 6 --warmup 2 --cuda-iters 10 --cuda-warmup 3
```

Artifacts:

- `/benchmarks/mamba3_cuda_full_bwd_ab/wave8_h200_ab_20260430_2/report.json`
- `/benchmarks/mamba3_cuda_full_bwd_ab/wave8_h200_ab_20260430_2/summary.json`
- `/benchmarks/mamba3_cuda_full_bwd_ab/wave8_h200_ab_20260430_2/summary.csv`

Device:

- `NVIDIA H200`
- Torch `2.13.0.dev20260426+cu132`
- Image `ghcr.io/jewelmusicee/cppmega:785c3fd`

## H200 TileLang AB

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | peak alloc GiB | peak reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke | baseline | 0.07972 | 0.16460 | 0.22537 | 0.00468 | 0.00586 |
| smoke | stage2_bf1_bb0 | 0.08122 | 0.16256 | 0.22862 | 0.00669 | 0.00781 |
| productionish | baseline | 1.89640 | 3.72332 | 5.59245 | 3.26655 | 3.28516 |
| productionish | stage2_bf1_bb0 | 1.81156 | 3.72232 | 5.50098 | 4.73024 | 4.74219 |

TileLang correctness:

| shape | comparison | max main grad abs diff |
| --- | --- | ---: |
| smoke | stage2_bf1_bb0 vs baseline | 0.0 |
| productionish | stage2_bf1_bb0 vs baseline | 0.0 |

Productionish stage2 `bwd_bwd` source metadata:

- source chars: `83354`
- sha256: `73a9c23d1ee10f27426d41fcf3c2950d50bb7eac8f573e9af8624f8ad8a7c7a2`
- launch bounds: `(256, 1)`
- TMA loads/stores: `0 / 0`
- WS producer guard: `false`

## H200 CUDA Components

| shape | component | launches | mean ms | peak alloc GiB | peak reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| smoke | wave6 diag | 1 | 0.02516 | 0.01145 | 0.01172 |
| smoke | wave7 qk/dV | 1 | 0.01297 | 0.01145 | 0.01172 |
| smoke | combined diag + qk/dV | 1 | 0.02525 | 0.01145 | 0.01172 |
| productionish | wave6 diag | 1 | 1.77763 | 6.92774 | 7.67969 |
| productionish | wave7 qk/dV | 1 | 0.36735 | 6.92774 | 7.67969 |
| productionish | combined diag + qk/dV | 1 | 1.92990 | 6.92774 | 7.67969 |

For the component-sum model, diag plus qk/dV as separate launches is:

| shape | two-launch component sum ms | combined one-launch ms | combined / sum |
| --- | ---: | ---: | ---: |
| smoke | 0.03812 | 0.02525 | 0.662 |
| productionish | 2.14498 | 1.92990 | 0.900 |

CUDA correctness summaries:

| shape | check | max abs diff |
| --- | --- | ---: |
| smoke | combined diag vs wave5 timestep-post CUDA | 5.684e-14 |
| smoke | combined dV vs torch qk/dV reference | 2.274e-13 |
| productionish | combined diag vs wave5 timestep-post CUDA | 9.095e-13 |
| productionish | combined dV vs torch qk/dV reference | 1.455e-11 |

CUDA kernel metadata on H200:

| kernel | regs/thread | local bytes | active blocks/SM | occupancy |
| --- | ---: | ---: | ---: | ---: |
| wave6 diag | 88 | 64 | 5 | 31.25% |
| wave7 qk/dV | 48 | 0 | 10 | 62.5% |
| wave7 combined | 80 | 64 | 6 | 37.5% |

## Replacement Math

This is an incomplete floor, not a full production replacement. The current
CUDA candidate covers:

- `DGAMMA_DIAG`;
- same-time diagonal contributions into `DK` and `DQ`;
- the same-time `qk_dot -> dPsiV -> DV` consumer.

It does not yet cover the full `bwd_bwd` output contract.

| shape | stage2 bwd_bwd ms | CUDA combined ms | ratio | floor speedup | remaining budget to equal TileLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 0.16256 | 0.02525 | 0.155 | 6.44x | 0.13731 ms |
| productionish | 3.72232 | 1.92990 | 0.518 | 1.93x | 1.79242 ms |

End-to-end floor if the incomplete CUDA candidate replaced all of `bwd_bwd`
without changing stage2 `bwd_fwd`:

| shape | stage2 chain ms | floor chain ms | floor speedup |
| --- | ---: | ---: | ---: |
| smoke | 0.22862 | 0.10647 | 2.15x |
| productionish | 5.50098 | 3.74146 | 1.47x |

Launch count model:

| path | bwd_bwd launch count | chain launch count |
| --- | ---: | ---: |
| TileLang stage2_bf1_bb0 | 1 | 2 |
| CUDA component sum | 2 | 3 if used after TileLang bwd_fwd |
| CUDA combined current candidate | 1 | 2 if used after TileLang bwd_fwd |

The important productionish number is the budget: after the current combined
CUDA slice, there is about `1.79 ms` left before matching the measured TileLang
stage2 `bwd_bwd`. That is the budget for all missing `bwd_bwd` work if the full
CUDA replacement is to be faster than TileLang on H200.

## H100:2 Smoke

Cheap H100 path was run and passed:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100:2 timeout 900s \
  modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
  --run-id wave8_h100_smoke_20260430_1 \
  --shape-csv smoke \
  --iters 2 --warmup 1 --cuda-iters 5 --cuda-warmup 1
```

Artifacts:

- `/benchmarks/mamba3_cuda_full_bwd_ab/wave8_h100_smoke_20260430_1/report.json`
- `/benchmarks/mamba3_cuda_full_bwd_ab/wave8_h100_smoke_20260430_1/summary.json`
- `/benchmarks/mamba3_cuda_full_bwd_ab/wave8_h100_smoke_20260430_1/summary.csv`

Device: `NVIDIA H100 80GB HBM3`, `device_count=2`.

| variant/component | mean ms |
| --- | ---: |
| TileLang baseline bwd_bwd | 0.16237 |
| TileLang stage2_bf1_bb0 bwd_bwd | 0.16110 |
| CUDA wave6 diag | 0.02536 |
| CUDA wave7 qk/dV | 0.01219 |
| CUDA combined diag + qk/dV | 0.02475 |

H100 smoke correctness matched the H200 smoke profile:

- TileLang stage2 vs baseline max main grad abs diff: `0.0`
- CUDA combined diag max abs diff: `5.684e-14`
- CUDA combined dV max abs diff: `2.274e-13`

## Readiness Criteria

Full CUDA `bwd_bwd` is not ready to replace production yet. Required before
replacement:

1. Port the missing off-time intra-chunk/state work.
2. Produce full `DK`, `DQ`, and `DV`, not only the same-time diagonal/qk-dV
   slices.
3. Implement `DMIMO_V` with a correct cross-chunk reduction or a different
   ownership model.
4. Produce and validate `dfactor`, `dangles`, `dd`, `dda`, `dssda`,
   `dda_cs_rev`, and `dda_cs`.
5. Compare the full CUDA outputs directly against TileLang stage2 for every
   output tensor on smoke, representative, and productionish shapes.
6. Integrate at the real `mamba_mimo_bwd_bwd` call boundary, not only as a
   standalone component harness.
7. Keep the launch count at one if possible; justify any split with measured
   speedup after memory traffic and allocation effects.
8. Recheck peak memory in the integrated path. The standalone CUDA component
   harness peaks higher than TileLang because it carries independent inputs,
   outputs, and references; this is acceptable for the harness but not a
   production memory claim.
9. Re-run H200 productionish, H100:2 smoke or representative, and B200 only if
   capacity is immediately available.

## Read

The full CUDA path is on track at the component-economics level, not at the
production-readiness level.

On H200 productionish, the current one-launch CUDA slice is `1.92990 ms` versus
TileLang stage2 `bwd_bwd` at `3.72232 ms`, leaving `1.79242 ms` of budget for
the missing full-kernel work before merely matching TileLang. That is a useful
budget, but the remaining work is the hard part: off-time/state math and
cross-chunk/reduction-owned outputs. The harness is now in place to score Lane
A/B kernel candidates end to end without NCU.
