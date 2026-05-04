# Mamba3 bwd_bwd Surface Reduction Wave 4 - 2026-04-29

Status: evidence
Canonical: none
Date: 2026-04-29
Scope: H200 Triton microbench for folded blockdiag productionish timing.

Branch: `worker/mamba3-bwd-bwd-owner-rewrite`

## Goal

Wave 3 proved that full `[F,F]` materialization is not mathematically required,
but the split streaming prototype was too slow on productionish:

| variant | productionish mean ms |
| --- | ---: |
| wave2 full-P DQ/DK/diag subproblem | `1.0865` |
| wave2 serial two-pass P_TILE=64 | `1.6625` |
| wave3 split streaming total | `5.3818` |

The open question was whether folding the pathological tiny `R x R` blockdiag
kernel back into the DK/DQ owner kernels made the streaming surface reduction
close enough to integrate.

## Files

- Baseline folded harness: `scripts/modal_mamba3_bwd_bwd_surface_reduction_wave3.py`
- Wave4 optimization harness: `scripts/modal_mamba3_bwd_bwd_surface_reduction_wave4.py`
- Status: `docs/status/mamba3_bwd_bwd_surface_reduction_wave4_2026_04_29.md`

Both harnesses are Triton/PyTorch microbenches, not production TileLang patches.
Shape specialization: `F=chunk*R=64`, `N=64`, `P=128`, `R=4`.

## Validation Environment

Modal H200 image:

- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- GPU: `NVIDIA H200`, capability `(9, 0)`, device count `2`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`
- Triton: `3.7.0`

Local:

```text
python -m py_compile scripts/modal_mamba3_bwd_bwd_surface_reduction_wave4.py
```

Modal runs:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 25m modal run \
  scripts/modal_mamba3_bwd_bwd_surface_reduction_wave3.py \
  --shape-csv productionish --warmup 2 --iters 8

CPPMEGA_MODAL_GPU=H200:2 timeout 25m modal run \
  scripts/modal_mamba3_bwd_bwd_surface_reduction_wave4.py \
  --shape-csv smoke_p128,productionish --warmup 2 --iters 8
```

## Correctness

Folded scalar productionish (`wave3` script, first 64 owners checked) passed
against the PyTorch full-surface reference:

| output | max abs |
| --- | ---: |
| DK | `2.7939677238464355e-09` |
| DQ | `3.259629011154175e-09` |
| DGAMMA | `4.3655745685100555e-11` |
| DSSDA | `7.275957614183426e-11` |

Wave4 row-vector folded optimization also passed smoke and productionish:

| output | productionish max abs |
| --- | ---: |
| DK | `2.6775524020195007e-09` |
| DQ | `3.259629011154175e-09` |
| DGAMMA | `4.3655745685100555e-11` |
| DSSDA | `7.275957614183426e-11` |

## Timing

Productionish shape: `B=4,S=4096,H=32,N=64,P=128,R=4,chunk=16`, `32768`
owners.

| variant | DK ms | DQ ms | total ms | vs wave2 full-P |
| --- | ---: | ---: | ---: | ---: |
| wave2 full-P DQ/DK/diag | n/a | n/a | `1.0865` | `1.00x` |
| wave3 split streaming | `1.2461` | `1.0982` | `5.3818` | `4.95x slower` |
| folded scalar blockdiag in owners | `5.9622` | `10.3513` | `16.3435` | `15.04x slower` |
| wave4 row-vector folded optimization | `10.1932` | `9.8395` | `20.0717` | `18.47x slower` |

The attempted direct `tl.dot(4x4,4x64)` fold did not compile because Triton
requires dot `K >= 16`. The row-vector fallback reduced full `F x N` masked
updates per `(dst,src)` pair, but still increased productionish runtime. The
fold did not rescue the path.

## Equations

Let:

```text
M_ij = <Psi_i, dPhi_j>
W_ij = causal_or_segment_weight(i,j)
```

For a live tile `I x J` of `M`, the three consumers are:

```text
DK_i    += sum_{j in J} W_ij * M_ij * Q_j       for i in I
DQ_j    += sum_{i in I} W_ij * M_ij * K_i       for j in J
DSSDA_i += sum_{j in J} qk_ij * M_ij            for i in I
```

Therefore DSSDA, DK, and DQ can share one live raw `M_{I,J}` tile exactly:
DSSDA consumes raw `M`, while DK/DQ consume the same tile after multiplying by
the same pair weight `W_ij`. No algebraic recompute is required.

The current split streaming prototype recomputes anyway because ownership is
split by output:

```text
DK owner: computes M_{I,J} while holding DK_I and DSSDA_I
DQ owner: computes M_{I,J} again while holding DQ_J
```

To avoid recompute, a kernel must own both `DK_I` and `DQ_J` updates for the
same `M_{I,J}` tile, or use atomics/partials. A single chunk owner with all
`DK` and `DQ` accumulators live is algebraically valid, but it returns to a
large accumulator footprint close to the wave2 full-P owner. A tile owner avoids
that footprint only with custom fused scheduling and controlled cross-tile
reductions.

For blockdiag:

```text
G_s = dPhi_s @ Psi_s.T = M_s.T for same-token R x R block s
DK_s += G_s.T @ Q_s
DQ_s += G_s @ K_s
```

So full `G` is not required either; the same-token `M_s` tile contains the
needed information. The measured Triton fold still loses because the scalar
`R=4` work is too small for efficient compiler lowering inside the already
expensive streaming owner.

## Decision

Current accumulator-surface reduction is dead as a standalone Lane C path. It
is correct, and full `[F,F]` is not mathematically required, but productionish
timing is nowhere near the wave2 full-P subproblem:

- Best folded productionish measured here: `16.3435 ms`.
- Prior split streaming productionish: `5.3818 ms`.
- Wave2 full-P productionish: `1.0865 ms`.

Do not integrate this Triton/TileLang split streaming approach into production.
Only revisit the algebra if paired with Lane A custom fused CUDA that can keep
one `M_{I,J}` tile live and feed DK, DQ, and DSSDA in the same scheduled tile.

## Wave5 Recommendation

Stop spending Lane C on accumulator-surface reduction. Wave5 should either:

1. Hand the one-live-`M` algebra to Lane A for a fused CUDA tile-owner design.
2. Return Lane C to the full-P owner path and optimize integration around the
   measured `1.0865 ms` baseline.

Avoid P_TILE atomics, serial two-pass, and split streaming as production
candidates unless Lane A supplies a fused kernel that removes recompute without
global partial handoff.

## Modal Cleanup

Wave4 apps launched here:

- `ap-lcmTT8H2PNLtfFrxKQQjw5` - stopped.
- `ap-FeTFogQv0AydDk37vCHRlt` - stopped.
- `ap-pqQDLUfmi5wyMlA2ii4jay` - stopped.

Final `modal app list --json` showed all
`cppmega-mamba3-bwd-bwd-surface-reduction-wave3` and
`cppmega-mamba3-bwd-bwd-surface-reduction-wave4` apps stopped. I did not stop
unrelated active apps from other lanes.
