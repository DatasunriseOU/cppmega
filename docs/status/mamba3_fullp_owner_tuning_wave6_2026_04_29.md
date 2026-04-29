# Mamba3 Full-P Owner Tuning Wave 6 - 2026-04-29

Status: evidence
Canonical: none
Date: 2026-04-29
Scope: H200 Triton microbench for monolithic full-P/full-M DQ/DK/diag ownership.

Branch: `worker/mamba3-fused-m-tile-owner`

Base before wave6 edits: `62fbf4d`

## Goal

Wave5 proved the no-atomic row owner with a global DQ partial handoff was
correct, but productionish `32x32` still took `2.2178 ms`, `2.041x` slower
than the wave2 full-P DQ/DK/diag subproblem (`1.0865 ms`). Wave6 returned to
the full-P owner and tuned only local ownership/compiler choices. No P_TILE
atomics, serial P_TILE handoff, or global partial reductions were used.

## Files

- Harness: `scripts/modal_mamba3_bwd_bwd_fullp_owner_tuning_wave6.py`
- Status: `docs/status/mamba3_fullp_owner_tuning_wave6_2026_04_29.md`

The harness keeps the same shape family as waves 2-5:
`F=chunk*R=64`, `N=64`, `P=128`, `R=4`.

## Variants

Reference wave2 semantics:

```text
DK         = Psi  @ dState
DQ         = dPhi @ State
G          = dPhi @ Psi.T
M          = Psi  @ dPhi.T
diag_qk    = row_sum(qk * G)
diag_intra = row_sum(qk * M)
```

Tuned variants:

- `full64_two_*`: exact wave2-style full owner, two F-by-F diag surface dots.
- `full64_one_*`: same outputs, but uses `M = G.T` and computes only one
  F-by-F diag surface dot.
- `row32_two_*` / `row16_two_*`: unique row-tile owners, still full-P and
  full-M columns for diag, no partials.
- `split*_diag1_*`: unique DQ/DK tile kernel plus a unique diag-only kernel,
  still no partials.

I did not implement an R=4-only diagonal shortcut for this wave2 subproblem
because the measured `1.0865 ms` reference uses full-row diag reductions. An
R=4-only shortcut would change this harness's semantics even though it may be
valid for narrower blockdiag consumers in the full stage2 kernel.

## Validation Environment

Modal H200 image:

- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- GPU: `NVIDIA H200`, capability `(9, 0)`, device count `2`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`
- Triton: `3.7.0`

Local:

```text
python -m py_compile scripts/modal_mamba3_bwd_bwd_fullp_owner_tuning_wave6.py
```

## Commands

Smoke subset:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_fullp_owner_tuning_wave6.py \
  --shape-csv smoke_p128 \
  --variant-csv full64_two_w8s3,full64_one_w8s3,row32_two_w4s3,split64x32_diag1_w4s3 \
  --warmup 1 --iters 3
```

App: `ap-vUsZujvAfglAvDXWoPWpy2`, stopped normally.

Smoke all variants:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_fullp_owner_tuning_wave6.py \
  --shape-csv smoke_p128 --warmup 1 --iters 3
```

App: `ap-vlP32CDyh8hTpe9sOd2pa8`, stopped normally.

Productionish broad sweep:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1500 modal run \
  scripts/modal_mamba3_bwd_bwd_fullp_owner_tuning_wave6.py \
  --shape-csv productionish --warmup 2 --iters 8
```

App: `ap-FXqOO5ILdJGQXhVVYNuRsb`, stopped normally.

Productionish focused repeat:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_fullp_owner_tuning_wave6.py \
  --shape-csv productionish \
  --variant-csv full64_two_w8s3,full64_one_w4s3,full64_one_w4s4,full64_one_w8s2,full64_one_w8s3,split32x64_diag1_w4s3 \
  --warmup 4 --iters 20
```

App: `ap-96g3SX3AH9MAR1nGU9dYoe`, stopped normally.

Productionish row-owner subset rerun:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_fullp_owner_tuning_wave6.py \
  --shape-csv productionish \
  --variant-csv row32_two_w4s3,row32_two_w8s3,row32_two_w4s2,row16_two_w4s3,row16_two_w8s3 \
  --warmup 2 --iters 8
```

App: `ap-TlgP6oaOwXSWLeGGUMnoTW`, stopped normally.

## Correctness

All reported variants passed against `full64_two_w8s3` at
`rtol=1e-2, atol=1e-2` for `DK`, `DQ`, `diag_qk`, and `diag_intra`.

Productionish focused repeat max abs for the best candidate
`full64_one_w4s4`:

| output | max abs vs full64_two_w8s3 |
| --- | ---: |
| `DK` | `0` |
| `DQ` | `0` |
| `diag_qk` | `0` |
| `diag_intra` | `3.64e-11` |

The nonzero `diag_intra` is from using `G.T` instead of a second explicit dot;
it is far below the existing tolerance.

## Smoke Sweep

Smoke shape: `B=1,S=256,H=4,N=64,P=128,R=4,chunk=16`, `64` owners,
`warmup=1,iters=3`. These timings are launch/noise dominated but caught compile
and correctness failures.

| variant | mode | mean ms | max abs note |
| --- | --- | ---: | --- |
| `full64_two_w8s3` | full/two-dot | `0.0334` | exact |
| `full64_two_w4s3` | full/two-dot | `0.0380` | exact |
| `full64_two_w8s2` | full/two-dot | `0.0459` | exact |
| `full64_two_w8s4` | full/two-dot | `0.0381` | exact |
| `full64_one_w8s3` | full/one-dot | `0.0507` | `diag_intra 4.37e-11` |
| `full64_one_w4s3` | full/one-dot | `0.0404` | `diag_intra 4.37e-11` |
| `full64_one_w8s2` | full/one-dot | `0.0383` | `diag_intra 4.37e-11` |
| `full64_one_w8s4` | full/one-dot | `0.0378` | `diag_intra 4.37e-11` |
| `full64_one_w4s4` | full/one-dot | `0.0372` | `diag_intra 4.37e-11` |
| `row32_two_w4s3` | row/two-dot | `0.0401` | exact |
| `row32_two_w8s3` | row/two-dot | `0.0378` | exact |
| `row32_two_w4s2` | row/two-dot | `0.0388` | exact |
| `row16_two_w4s3` | row/two-dot | `0.0382` | exact |
| `row16_two_w8s3` | row/two-dot | `0.0383` | `diag max 5.82e-11` |
| `split64x32_diag1_w4s3` | split/one-dot diag | `0.0365` | `diag_intra 4.37e-11` |
| `split64x32_diag1_w8s3` | split/one-dot diag | `0.0376` | `diag_intra 4.37e-11` |
| `split32x32_diag1_w4s3` | split/one-dot diag | `0.0349` | `diag_intra 4.37e-11` |
| `split32x64_diag1_w4s3` | split/one-dot diag | `0.0337` | `diag_intra 4.37e-11` |

## Productionish Sweep

Productionish shape: `B=4,S=4096,H=32,N=64,P=128,R=4,chunk=16`,
`32768` owners.

Broad productionish table, using `warmup=2,iters=8`; row entries are from the
row subset rerun with the same settings because the first broad CLI log was
truncated around that section.

| variant | kernels | programs | diag dots | mean ms | vs wave2 `1.0865` | vs stage2 `3.6940` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full64_two_w8s3` | 1 | `32768` | 2 | `1.1225` | `1.033x` | `0.304x` |
| `full64_two_w4s3` | 1 | `32768` | 2 | `1.1325` | `1.042x` | `0.307x` |
| `full64_two_w8s2` | 1 | `32768` | 2 | `1.1274` | `1.038x` | `0.305x` |
| `full64_two_w8s4` | 1 | `32768` | 2 | `1.1324` | `1.042x` | `0.307x` |
| `full64_one_w8s3` | 1 | `32768` | 1 | `1.1032` | `1.015x` | `0.299x` |
| `full64_one_w4s3` | 1 | `32768` | 1 | `1.0805` | `0.994x` | `0.293x` |
| `full64_one_w8s2` | 1 | `32768` | 1 | `1.0912` | `1.004x` | `0.295x` |
| `full64_one_w8s4` | 1 | `32768` | 1 | `1.1008` | `1.013x` | `0.298x` |
| `full64_one_w4s4` | 1 | `32768` | 1 | `1.0812` | `0.995x` | `0.293x` |
| `row32_two_w4s3` | 1 | `65536` | 2 | `1.6674` | `1.535x` | `0.451x` |
| `row32_two_w8s3` | 1 | `65536` | 2 | `1.6079` | `1.480x` | `0.435x` |
| `row32_two_w4s2` | 1 | `65536` | 2 | `1.6672` | `1.534x` | `0.451x` |
| `row16_two_w4s3` | 1 | `131072` | 2 | `2.3421` | `2.156x` | `0.634x` |
| `row16_two_w8s3` | 1 | `131072` | 2 | `2.7090` | `2.493x` | `0.733x` |
| `split64x32_diag1_w4s3` | 2 | `65536+32768` | 1 | `1.3807` | `1.271x` | `0.374x` |
| `split64x32_diag1_w8s3` | 2 | `65536+32768` | 1 | `1.3799` | `1.270x` | `0.374x` |
| `split32x32_diag1_w4s3` | 2 | `131072+32768` | 1 | `1.8018` | `1.658x` | `0.488x` |
| `split32x64_diag1_w4s3` | 2 | `65536+32768` | 1 | `1.3666` | `1.258x` | `0.370x` |

Focused productionish repeat, `warmup=4,iters=20`:

| variant | mean ms | min ms | max ms | vs wave2 `1.0865` | same-run vs `full64_two_w8s3` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full64_two_w8s3` | `1.1247` | `1.1137` | `1.1359` | `1.035x` | `1.000x` |
| `full64_one_w4s3` | `1.0828` | `1.0522` | `1.1133` | `0.997x` | `0.963x` |
| `full64_one_w4s4` | `1.0803` | `1.0592` | `1.1080` | `0.994x` | `0.960x` |
| `full64_one_w8s2` | `1.0949` | `1.0761` | `1.1102` | `1.008x` | `0.974x` |
| `full64_one_w8s3` | `1.0911` | `1.0780` | `1.1058` | `1.004x` | `0.970x` |
| `split32x64_diag1_w4s3` | `1.3643` | `1.3570` | `1.3731` | `1.256x` | `1.213x` |

## Readout

The only useful tuning result is the one-dot full owner:

- It removes one F-by-F tensor-core dot by recognizing
  `Psi @ dPhi.T == (dPhi @ Psi.T).T`.
- It lowers the modeled accumulator footprint from `139,776` to `123,392`
  bytes per owner program.
- It is `~4.0%` faster than the same-run two-dot full owner in the focused
  repeat.
- Against the prior wave2 productionish reference (`1.0865 ms`), the best
  focused value is only `1.0803 ms`, a `0.6%` improvement.

Everything that reduced accumulator pressure by increasing owner count lost:

- `row32` is `1.48-1.53x` slower than wave2 because it doubles programs and
  rereads full-M columns for diag.
- `row16` is `2.16-2.49x` slower.
- Split DQ/DK plus diag avoids partials but pays another launch and extra
  input traffic; best split was `1.3643 ms`, `1.26x` slower than wave2.

## Decision

A tuned full-P owner survives, but barely: `full64_one_w4s4` is the only
candidate worth carrying forward, and it is a local algebra simplification,
not a new ownership design. The measured margin versus the old `1.0865 ms`
baseline is too small to justify a larger rewrite by itself.

Next wave recommendation:

1. Inspect the production TileLang bwd_bwd owner for duplicated
   `dPhi @ Psi.T` / `Psi @ dPhi.T` surfaces and replace the second surface with
   a transpose only where both consumers already have the full surface live.
2. Benchmark the full stage2 kernel after that local change; the expected
   impact is small, likely sub-1% end-to-end on stage2 unless it also reduces
   register pressure enough to change occupancy.
3. Do not continue row-tiled or two-kernel split variants for this subproblem.
   They repeat the wave5 pattern: correct, no atomics, but more programs or
   global traffic than the monolithic owner.

## Modal Cleanup

Wave6 apps launched by this work:

- `ap-vUsZujvAfglAvDXWoPWpy2` - stopped.
- `ap-vlP32CDyh8hTpe9sOd2pa8` - stopped.
- `ap-FXqOO5ILdJGQXhVVYNuRsb` - stopped.
- `ap-96g3SX3AH9MAR1nGU9dYoe` - stopped.
- `ap-TlgP6oaOwXSWLeGGUMnoTW` - stopped.

Final `modal app list --json` showed all
`cppmega-mamba3-bwd-bwd-fullp-owner-tuning-wave6` apps stopped. It also showed
unrelated active apps (`cppmega-mamba3-stage2-profile-wave6` and
`cppmega-mamba3-stage2-force-nontma-benchmark`) that this work did not launch
and did not stop.
