# Mamba3 Fused M-Tile Owner Wave 5 - 2026-04-29

Status: evidence
Canonical: none
Date: 2026-04-29
Scope: H200 Triton microbench for one-live-`M` tile ownership in `bwd_bwd`.

Branch: `worker/mamba3-fused-m-tile-owner`

## Goal

Wave2 showed narrow `P_TILE` atomics and serial two-pass ownership were slower
than the full-P DQ/DK/diag subproblem. Wave3/Wave4 showed the surface algebra
is correct, but split streaming recomputed `M` for DK and DQ and was too slow.

Wave5 tested the missing ownership shape: form one live
`M_{I,J} = Psi_I @ dPhi_J.T` tile and consume it for DK, DQ, DSSDA, and
DGAMMA while the tile is live.

## Files

- Harness: `scripts/modal_mamba3_bwd_bwd_fused_m_tile_owner_wave5.py`
- Status: `docs/status/mamba3_fused_m_tile_owner_wave5_2026_04_29.md`

The harness is a Triton microbench, not a production TileLang patch. It
specializes the same stage2 probe shape family used by prior waves:
`F=chunk*R=64`, `N=64`, `P=128`, `R=4`.

## Design

Reference:

```text
M = Psi @ dPhi.T
DK += (M * W) @ Q + blockdiag(M) @ Q
DQ += (M * W).T @ K + blockdiag(M).T @ K
DGAMMA = row_sum(qk * blockdiag(M).T)
DSSDA  = row_sum(qk * M)
```

The row-owner candidate uses no atomics:

1. One Triton program owns an `I` row tile for one `(B,H,chunk)` owner.
2. It loops over `J` tiles.
3. For each `J`, it forms one live `M_{I,J}` tile.
4. It immediately accumulates owned `DK_I`, `DSSDA_I`, and `DGAMMA_I`.
5. It emits a `DQ_J` partial for this `I` tile into a unique partial buffer.
6. A second reducer kernel owns final `DQ_J` and reduces partials over `I`.

This removes the wave4 DK/DQ recomputation of `M`, but it pays a non-atomic DQ
partial handoff. Square tile sizes were tested:

- `16x16`: four `I` blocks, DQ partial buffer is `4x` final DQ.
- `32x32`: two `I` blocks, DQ partial buffer is `2x` final DQ.

## Validation Environment

Modal H200 image:

- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- GPU: `NVIDIA H200`, capability `(9, 0)`, device count `2`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`
- Triton: `3.7.0`

Local:

```text
python -m py_compile scripts/modal_mamba3_bwd_bwd_fused_m_tile_owner_wave5.py
```

Prior-wave context inspected:

- `scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave2.py`
- `scripts/modal_mamba3_bwd_bwd_surface_reduction_wave3.py`
- `scripts/modal_mamba3_bwd_bwd_surface_reduction_wave4.py`
- `docs/status/mamba3_bwd_bwd_owner_rewrite_wave2_2026_04_29.md`
- `docs/status/mamba3_bwd_bwd_surface_reduction_wave3_2026_04_29.md`
- `docs/status/mamba3_bwd_bwd_surface_reduction_wave4_2026_04_29.md`
- `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py`

## Commands

Smoke, `16x16`:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_fused_m_tile_owner_wave5.py \
  --shape-csv smoke_p128 --warmup 1 --iters 3
```

App: `ap-5nmWUf6pVRAGKcDvw2Vpev`, stopped normally.

Productionish, `16x16`:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1500 modal run \
  scripts/modal_mamba3_bwd_bwd_fused_m_tile_owner_wave5.py \
  --shape-csv representative,productionish --warmup 2 --iters 8
```

App: `ap-RuzfejY3E70Z7vDzEUXinW`, stopped normally.

Smoke, `32x32`:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_fused_m_tile_owner_wave5.py \
  --shape-csv smoke_p128 --warmup 1 --iters 3 \
  --block-i 32 --block-j 32
```

App: `ap-tQXRy4GoBH38YfpNb2hgk8`, stopped normally.

Productionish, `32x32`:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1500 modal run \
  scripts/modal_mamba3_bwd_bwd_fused_m_tile_owner_wave5.py \
  --shape-csv representative,productionish --warmup 2 --iters 8 \
  --block-i 32 --block-j 32
```

App: `ap-WToPYH5QxbvZhdGG4SjG2Z`, stopped normally.

## Correctness

All variants passed against the full-`M` surface reference at
`rtol=1e-2, atol=1e-2`. The full-`M` surface reference also passed against the
PyTorch surface reference.

Productionish `16x16`, first 64 owners checked:

| comparison | DK max abs | DQ max abs | DGAMMA max abs | DSSDA max abs |
| --- | ---: | ---: | ---: | ---: |
| full `M` surface vs PyTorch | `3.35e-09` | `3.45e-09` | `2.18e-11` | `7.28e-11` |
| row owner vs full `M` surface | `0` | `1.46e-11` | `0` | `5.82e-11` |

Productionish `32x32`, first 64 owners checked:

| comparison | DK max abs | DQ max abs | DGAMMA max abs | DSSDA max abs |
| --- | ---: | ---: | ---: | ---: |
| full `M` surface vs PyTorch | `3.35e-09` | `3.45e-09` | `2.18e-11` | `7.28e-11` |
| row owner vs full `M` surface | `0` | `1.46e-11` | `0` | `4.37e-11` |

## Timing

Existing productionish baselines:

| baseline | mean ms |
| --- | ---: |
| wave2 full-P DQ/DK/diag subproblem | `1.0865` |
| stage2 `bwd_bwd` default `(bf=1,bb=0)` full kernel | `3.6940` |

Wave5 productionish:

| tile | full `M` surface ms | row-owner compute ms | DQ reduce ms | row-owner total ms | row total vs full `M` | row total vs wave2 | row total vs stage2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `16x16` | `1.7345` | `2.5355` | `0.6519` | `3.1635` | `1.824x slower` | `2.912x slower` | `0.856x` |
| `32x32` | `1.7229` | `1.8396` | `0.3939` | `2.2178` | `1.287x slower` | `2.041x slower` | `0.600x` |

Representative:

| tile | full `M` surface ms | row-owner compute ms | DQ reduce ms | row-owner total ms | row total vs full `M` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `16x16` | `0.0890` | `0.1127` | `0.0442` | `0.1364` | `1.532x slower` |
| `32x32` | `0.0758` | `0.0817` | `0.0254` | `0.0928` | `1.224x slower` |

## Memory Model

Productionish `B=4,S=4096,H=32,N=64,P=128,R=4,chunk=16`, `32768` owners:

| tile | row-owner programs | reducer programs | DQ partial write | DQ partial read | final DQ write | partial multiplier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `16x16` | `131072` | `131072` | `2147483648 B` | `2147483648 B` | `536870912 B` | `4.0x` |
| `32x32` | `65536` | `65536` | `1073741824 B` | `1073741824 B` | `536870912 B` | `2.0x` |

The `32x32` tile is the better point because it halves the partial handoff and
program count while still keeping only a half-surface `M` tile live. Even there,
the row-owner compute kernel alone is slightly slower than the monolithic
full-`M` surface, and the DQ partial handoff adds another `0.3939 ms`.

## Decision

The algebra survives; this no-atomic Triton row-owner implementation does not
survive as a production direction.

Useful result:

- A single live `M_{I,J}` tile can feed DK, DQ, DSSDA, and DGAMMA correctly.
- Avoiding atomics by handing off DQ partials is also correctness-clean.

Blocking result:

- The best measured tile owner, `32x32`, is `2.2178 ms` on productionish for
  this surface subproblem.
- That is `1.287x` slower than the paired monolithic full-`M` surface
  (`1.7229 ms`) and `2.041x` the wave2 full-P DQ/DK/diag subproblem
  (`1.0865 ms`).
- The subproblem being `0.600x` of the full stage2 `bwd_bwd` time is not a
  win, because this harness omits other `bwd_bwd` work and still must be
  integrated.

Do not port the Triton row-owner + global DQ partial reduction into TileLang as
the next implementation. It repeats the same pattern as earlier dead paths:
the math is right, but ownership creates either extra programs, global partial
roundtrips, or both.

## Next Wave Recommendation

Only continue the one-live-`M` direction if the next wave can remove the global
DQ partial handoff, for example with a custom CUDA/CuTe schedule that performs
cross-`I` DQ reduction inside a CTA/cluster/persistent owner before final
global stores. Without that, return to the monolithic full-P/full-`M` owner and
optimize live-set pressure locally.

The concrete next experiment, if any, should be custom CUDA rather than another
Triton split: one owner group per `(B,H,chunk,J)` final DQ tile cooperatively
reduces multiple `I` tiles while a paired DK owner path keeps DK row ownership.
If that still needs global partials or atomics, stop the lane.

## Modal Cleanup

Wave5 apps launched here:

- `ap-5nmWUf6pVRAGKcDvw2Vpev` - stopped.
- `ap-RuzfejY3E70Z7vDzEUXinW` - stopped.
- `ap-tQXRy4GoBH38YfpNb2hgk8` - stopped.
- `ap-WToPYH5QxbvZhdGG4SjG2Z` - stopped.

Final `modal app list --json` showed the wave5 apps stopped. It also showed
unrelated running apps, including `cppmega-mamba3-stage2-force-nontma-benchmark`
`ap-Ml0QbF58Bux9ZWrsak2YNj` and `cppmega-mamba3-stage2-cuda-ab-benchmark`
`ap-osqdAOlwWvl1hiSCtDhHTU`, which this wave did not launch and did not stop.
