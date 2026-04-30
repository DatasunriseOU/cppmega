# Mamba3 Monolithic Triton Reuse Wave1 - 2026-04-30

Status: evidence / design guide
Canonical: no
Branch: `worker/mamba3-mono-triton-model`

## Scope

Lane C asked whether a monolithic algebra owner can quantify reuse across
`DV`, `DMIMO_V`, `DK/DQ`, and scalar outputs instead of producing another
standalone state/LKQ/D slice.

I inspected:

- `docs/status/mamba3_state_lkq_d_tensor_wave10_2026_04_30.md`
- `docs/status/mamba3_cuda_remaining_state_wave9_2026_04_30.md`
- `docs/status/mamba3_cuda_dmimo_reduce_wave8_2026_04_30.md`
- `docs/status/mamba3_rr_diag_cuda_integrate_wave7_2026_04_30.md`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave10_state_lkq_d_triton.py`
- `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py`

Prior state:

- Wave8 combined fast CUDA subset: `2.45093 ms`.
- Wave10 isolated Triton state/LKQ/D: `2.86062 ms`.
- Wave10 executed `42.95B` FMA for `29.26B` useful causal FMA.
- TileLang full `stage2_bf1_bb0` `bwd_bwd`: `3.70674 ms`.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave1_mono_triton_reuse_model.py`
- `scripts/modal_mamba3_mono_triton_reuse_wave1.py`

The probe is a monolithic owner-body model, not an output-correct full kernel.
It has two paths:

1. A materialized torch checksum reference for smoke correctness.
2. A Triton checksum microkernel with one program per `(B,H,chunk)` and one
   full `P=128` tile.

The Triton path computes these intermediates once per chunk owner:

- `LKQ = K @ Q.T`, reused by `dPsiV` and `DSSDA`.
- `dk_intra = PsiV @ dPhiO.T`, reused by `DGAMMA_DIAG`, `DSSDA`, `DK`, and `DQ`.
- `K @ dstates`, `PsiV @ dstates.T`, and `dPhiO @ states.T` state products.
- `qk_dot -> dPsiV` once, feeding both `DV` and `DMIMO_V` checksum paths.

The Triton kernel stores only one checksum per chunk. Timing is therefore a
compute/reuse lower bound; output traffic is reported by the model.

## H200 Runs

Smoke correctness/timing:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_triton_reuse_wave1.py \
  --shape-csv smoke --num-warps-csv 4 \
  --iters 5 --warmup 2 --check-torch-shapes smoke
```

Productionish timing and warp sweep:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_triton_reuse_wave1.py \
  --shape-csv productionish --num-warps-csv 4,8 \
  --iters 5 --warmup 2 --check-torch-shapes ''
```

Runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

## Correctness

Smoke shape: `B=1, S=256, H=4, N=64, P=128, R=4, chunk=16`.

| check | value |
| --- | ---: |
| torch checksum finite | `true` |
| Triton checksum finite | `true` |
| Triton vs torch checksum max abs delta | `6.083e-04` |
| Triton vs torch checksum mean abs delta | `1.390e-04` |
| max reference checksum abs | `2.243` |

This is only a checksum comparison. It is enough for the algebra probe but not
a replacement for output-wise full-kernel validation.

## Operation Model

Productionish shape: `B=4, S=4096, H=32, N=64, P=128, R=4, chunk=16`.

| model | FMA |
| --- | ---: |
| Separate-slice recompute model | `125.37B` |
| Monolithic reuse, full masked dots | `114.63B` |
| Monolithic reuse, triangular/causal applies pruned | `96.38B` |

Reuse alone saves:

- `10.74B` FMA (`8.56%`) by avoiding duplicate `LKQ`, duplicate qk `DMIMO_V`
  recompute, and duplicate `dqk` diagonal work.
- `28.99B` FMA (`23.12%`) if the monolithic CUDA design also avoids full
  lower-triangle masked dot work.

The isolated Wave10 state/LKQ/D slice remains the key caution:

| Wave10 state/LKQ/D item | FMA |
| --- | ---: |
| executed full-mask Triton work | `42.95B` |
| useful causal work | `~29.26B` |

Monolithic reuse fixes duplicate consumers, but full masked `64x64` dot work is
still too expensive unless the CUDA design can prune or restructure it.

## Memory Model

Productionish estimated global traffic / storage:

| item | MiB |
| --- | ---: |
| `DV` output | `128.0` |
| `DK` output | `256.0` |
| `DQ` output | `256.0` |
| `DMIMO_V` per-chunk partials | `64.0` |
| scalar outputs including `DSSDA` and `DANGLES` | `74.0` |
| required monolithic output writes | `778.25` |
| final `DMIMO_V` reduce extra R/W | `128.25` |

Global temporaries avoided by a real chunk owner:

| avoided temp | MiB |
| --- | ---: |
| `dPsiV` bf16 temp | `512.0` |
| `LKQ` fp32 temp | `512.0` |
| `dk_intra` fp32 temp | `512.0` |
| state `dPsiV` fp32 temp | `1024.0` |
| `DK` fp32 temp | `512.0` |
| `DQ` fp32 temp | `512.0` |

So the memory win is real: a monolithic CUDA owner avoids roughly `3.5 GiB` of
materialized intermediates, but it still has about `0.76 GiB` of real output
writes plus the `DMIMO_V` final reduce traffic.

## Timings

Triton checksum lower bound, no full output stores:

| shape | warps | mean ms | min ms | full-mask TFMA/s |
| --- | ---: | ---: | ---: | ---: |
| smoke | 4 | `0.08913` | `0.07859` | `2.51` |
| productionish | 4 | `4.53881` | `4.53267` | `25.26` |
| productionish | 8 | `5.39902` | `5.38995` | `21.23` |

The previous 8-warp smoke run was faster (`0.05718 ms`) because the shape is
underfilled/noisy; productionish is the decision point, and 4 warps wins there.

At the measured 4-warp production throughput:

- Full-mask monolithic compute lower bound: `4.54 ms`.
- If causal applies were perfectly pruned at the same throughput:
  `96.38B / 25.26 TFMA/s ~= 3.82 ms`.
- This excludes `DV/DK/DQ/scalar` stores and the `DMIMO_V` reducer.

## Read

Triton can guide the CUDA monolithic design, but it is not the implementation
vehicle for this shape as written.

What it shows clearly:

1. Reuse is measurable but not enough by itself: `8.6%` FMA saved from duplicate
   consumers.
2. The bigger lever is triangular work removal: reuse plus causal pruning gets
   to `23.1%` FMA saved versus separate recompute.
3. A real owner should compute `LKQ` and `dk_intra` once, keep them in
   shared/register storage, and feed `DV`, combined `DMIMO_V` partials,
   `DK/DQ`, `DSSDA`, `DDA_CS`, `DDA_CS_REV`, `DFACTOR`, and `DGAMMA_DIAG`.
4. Triton full-block ownership is already slower than TileLang before output
   stores (`4.54 ms` vs `3.71 ms`), so the CUDA path needs CuTe/CUDA-level
   scheduling, triangular pruning, and tighter output staging.

Conclusion: **yes, Triton can guide the CUDA monolithic algebra and quantify
reuse, but no, this Triton owner body is not a viable replacement kernel.**
The CUDA design should use this model as the lower-bound ledger: target less
than `96B` effective FMA, avoid the `3.5 GiB` intermediate materialization, and
budget the unavoidable `~906 MiB` output/reducer traffic.
