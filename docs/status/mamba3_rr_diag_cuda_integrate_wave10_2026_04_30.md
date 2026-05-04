# Mamba3 R x R Diagonal CUDA Integration Wave 10 - 2026-04-30

Branch: `worker/mamba3-rr-diag-cuda-integrate`

## Goal

Tune the current covered CUDA subset from wave9 without adding missing state
work.  The covered subset remains:

- `DGAMMA_DIAG`
- diagonal `DK` / `DQ`
- qk-dot same-time `DV`
- qk-dot same-time `DMIMO_V`

State/LKQ/D and non-diagonal `DK` / `DQ` are still outside this prototype.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_chunk_owner_cuda.py`
- `scripts/modal_mamba3_rr_diag_wave10_cuda_tuning.py`
- `docs/status/mamba3_rr_diag_cuda_integrate_wave10_2026_04_30.md`

Changes:

- Made CUDA tuning compile-time selectable through:
  - `RR_DIAG_THREADS` (default now `256`)
  - `RR_DIAG_DMIMO_P_TILE` (default `32`)
  - `RR_DIAG_DMIMO_UNROLL` (default `1`)
  - `RR_DIAG_DMIMO_BROADCAST_QK` (default `0`)
- Generalized chunk-warp batching from fixed 4 warps to `kThreads / 32`.
- Generalized all-R output-owner `DMIMO_V` over `P_TILE`, while keeping the
  tuned default at `P_TILE=32`.
- Added a two-launch composition timing:
  `wave7 diag+qk/dV` launch plus output-owner all-R `DMIMO_V` launch.
- Added a wave10 Modal runner that can sweep variants without editing source.

## Local Checks

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_chunk_owner_cuda.py \
  scripts/modal_mamba3_rr_diag_wave10_cuda_tuning.py

git diff --check
```

Both passed.

## H200 Sweep

Command:

```text
timeout 2400s modal run --timestamps \
  scripts/modal_mamba3_rr_diag_wave10_cuda_tuning.py \
  --shape-csv productionish \
  --variants-csv t128p32u1,t128p64u1,t128p128u1,t128p32u2,t128p32u4,t64p32u1,t256p32u1,t128p32u1b1 \
  --warmup 2 \
  --iters 5
```

Productionish shape:
`B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| variant | qk/`DMIMO_V` mean ms | one-launch combined mean ms | two-launch mean ms | read |
| --- | ---: | ---: | ---: | --- |
| `t128p32u1` | `0.53459` | `2.48003` | `2.45418` | wave9-class baseline |
| `t128p64u1` | `1.03038` | `2.87628` | `2.94569` | larger P tile loses |
| `t128p128u1` | `2.02665` | `3.88864` | `3.95668` | not viable |
| `t128p32u2` | `0.54447` | `2.49128` | `2.48168` | unroll reg/ILP tradeoff loses |
| `t128p32u4` | `0.54429` | `2.47508` | `2.46381` | no slice gain |
| `t64p32u1` | `1.01923` | `3.02129` | `3.11066` | too few warps |
| `t256p32u1` | `0.30947` | `2.32460` | `2.10148` | winner |
| `t128p32u1b1` | `0.80627` | `2.63539` | `2.72983` | qk/gamma warp broadcast loses |

## Focused H200 Repeat

Command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_rr_diag_wave10_cuda_tuning.py \
  --shape-csv smoke,productionish \
  --variants-csv t256p32u1 \
  --warmup 5 \
  --iters 20
```

Device/runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

Productionish focused timings:

| path | mean ms | std ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wave7 diag + qk/`DV` combined, `threads=256` | `1.79059` | `0.00199` | `1.78704` | `1.79104` | `1.79395` |
| qk/`DMIMO_V` all-R output-owner, `threads=256` | `0.30854` | `0.00256` | `0.30627` | `0.30758` | `0.31776` |
| one-launch combined, `threads=256` | `2.31212` | `0.00318` | `2.30726` | `2.31126` | `2.32022` |
| two-launch wave7 + qk/`DMIMO_V`, `threads=256` | `2.09673` | `0.00330` | `2.09318` | `2.09658` | `2.10874` |
| TileLang `stage2_bf1_bb0` `bwd_bwd` baseline | `3.70674` | n/a | n/a | n/a | n/a |

Best stable covered-subset number from this wave:

- `2.09673 ms` mean for the two-launch composition.
- `2.31212 ms` mean for the best one-launch combined path.

Margin:

- two-launch is `56.6%` of TileLang and leaves `1.61001 ms` margin.
- one-launch is `62.4%` of TileLang and leaves `1.39462 ms` margin.
- qk/`DMIMO_V` slice improved from wave9 `0.53122 ms` to `0.30854 ms`.

Smoke focused timings were also correct; the smoke shape is underfilled and
therefore not used for final throughput claims.

## Correctness

Focused H200 productionish max absolute diffs:

| output | check | max abs diff |
| --- | --- | ---: |
| `DGAMMA_DIAG` | combined vs wave5 CUDA post reference | `7.105e-15` |
| `DK` | combined vs wave5 CUDA post reference | `9.095e-13` |
| `DQ` | combined vs wave5 CUDA post reference | `4.547e-13` |
| `DV` | combined vs torch qk/`DV` reference | `1.455e-11` |
| `DMIMO_V` | combined vs torch qk/`DMIMO_V` reference | `3.730e-14` |
| `DMIMO_V` | sequence-owner vs output-owner | `9.948e-14` |

Focused H200 smoke max absolute diffs:

- `DGAMMA_DIAG`: `1.776e-15`
- `DK`: `2.842e-14`
- `DQ`: `5.684e-14`
- `DV`: `2.274e-13`
- `DMIMO_V`: `2.665e-15`

## Resource Metadata

Focused H200 `t256p32u1` metadata:

| kernel | threads | regs/thread | static smem | active blocks/SM | occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| qk/`DMIMO_V` all-R output-owner | 256 | 40 | 4096 B | 6 | 75.0% |
| one-launch combined all-R owner | 256 | 80 | 4096 B | 3 | 37.5% |
| wave7 diag + qk/`DV` combined | 256 | 80 | 0 B | 3 | 37.5% |
| qk/`DV` chunk-warp slice | 256 | 56 | 0 B | 4 | 50.0% |
| sequence-owner qk/`DMIMO_V` | 256 | 48 | 0 B | 5 | 62.5% |

The two-launch composition wins because the qk/`DMIMO_V` kernel runs with its
own lower register footprint and 75% theoretical occupancy instead of sharing
the one-launch combined kernel's 80-register envelope.

## H100 Smoke

An H100 smoke was started after the H200 focused run:

```text
env CPPMEGA_MODAL_GPU=H100 modal run --timestamps \
  scripts/modal_mamba3_rr_diag_wave10_cuda_tuning.py \
  --shape-csv smoke \
  --variants-csv t256p32u1 \
  --warmup 2 \
  --iters 5
```

Per the follow-up instruction to stop expanding the sweep, it was stopped during
extension build with:

```text
modal app stop -y ap-AiS1lKuxdR9D9Z5PnyR1Q4
```

No H100 timing was collected.

## App Cleanup

`modal app list` after cleanup showed all wave10 apps stopped with `0` tasks.
Only the unrelated pre-existing deployed app remained.

## Read

The current covered subset has a solid speed margin on H200.  The best stable
number for the covered subset is the two-launch composition at `2.09673 ms`,
which is `1.61001 ms` faster than the current TileLang `bwd_bwd` comparison.
If a single CUDA launch is required, the tuned one-launch path is still solid at
`2.31212 ms`, retaining `1.39462 ms` margin.

The main tuning result is counterintuitive but clear: increasing the block to
256 threads improves both the all-R output-owner `DMIMO_V` slice and the
wave7 combined body enough to offset lower block residency.  Larger `P_TILE`,
extra unroll, 64-thread blocks, and warp-broadcasted qk/gamma loads all lose.
