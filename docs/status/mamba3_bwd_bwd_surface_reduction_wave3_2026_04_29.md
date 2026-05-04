# Mamba3 bwd_bwd Surface Reduction Wave 3 - 2026-04-29

Status: cutoff evidence
Canonical: none
Date: 2026-04-29
Scope: H200 Triton microbench for reducing the bwd_bwd `[F,F]` accumulator surface at stage2 shapes.

Branch: `worker/mamba3-bwd-bwd-owner-rewrite`

## Goal

Wave 1/2 showed P-tiling ownership is correct but slower at production fanout.
Wave 3 pivoted to algebraic surface reduction: avoid materializing full
`dqk_from_diag` and `dk_intrachunk` `[F,F]` accumulators where outputs only need
matrix-vector products or block-local reductions.

## Files

- Harness: `scripts/modal_mamba3_bwd_bwd_surface_reduction_wave3.py`
- Status: `docs/status/mamba3_bwd_bwd_surface_reduction_wave3_2026_04_29.md`

The harness is a Triton/PyTorch microbench, not a production TileLang patch.
Shape specialization: `F=chunk*R=64`, `N=64`, `P=128`, `R=4`.

## Equations Tried

Full reference surfaces:

```text
G = dPhi @ Psi.T          # dqk_from_diag
M = Psi @ dPhi.T          # dk_intrachunk
W_ij = causal(i,j) * weight_ij

DK = Psi @ dState.T + (M * W) @ Q + blockdiag(G).T @ Q
DQ = dPhi @ State.T + (M * W).T @ K + blockdiag(G) @ K
DGAMMA_i = sum_j qk_ij * blockdiag(G)_ij
DSSDA_i  = sum_j qk_ij * M_ij
```

Streaming candidate:

```text
DK_i += sum_j W_ij * <Psi_i, dPhi_j> * Q_j
DQ_j += sum_i W_ij * <Psi_i, dPhi_j> * K_i
DSSDA_i += sum_j qk_ij * <Psi_i, dPhi_j>
```

For `dqk_from_diag`, the only consumed part is same-token `R x R`:

```text
G_s = dPhi_s @ Psi_s.T, where s is one chunk token and G_s is 4x4
DK_s += G_s.T @ Q_s
DQ_s += G_s @ K_s
DGAMMA_s = row_sum(qk_s * G_s)
```

Unweighted associative form, noted but not valid for the masked production
term:

```text
M @ Q   = Psi  @ (dPhi.T @ Q)
M.T @ K = dPhi @ (Psi.T  @ K)
```

This avoids `[F,F]` only when `W_ij` is absent or separable. The causal/segment
weight is pair-dependent, so it cannot be factored into a single `[P,N]`
pre-reduction.

## Validation Environment

Modal H200 image:

- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- GPU: `NVIDIA H200`, capability `(9, 0)`, device count `2`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`
- Triton: `3.7.0`

Local:

```text
python -m py_compile scripts/modal_mamba3_bwd_bwd_surface_reduction_wave3.py
```

## Correctness

Smoke (`B=1,S=256,H=4`, 64 owners) passed against PyTorch full-surface
reference for all outputs:

| output | max abs |
| --- | ---: |
| DK | `9.96e-08` |
| DQ | `1.04e-07` |
| DGAMMA | `2.91e-11` |
| DSSDA | `1.02e-10` |

Productionish (`B=4,S=4096,H=32`, first 64 owners checked) passed before the
cutoff with the split three-kernel prototype:

| output | max abs |
| --- | ---: |
| DK | `2.68e-09` |
| DQ | `3.26e-09` |
| DGAMMA | `4.37e-11` |
| DSSDA | `7.28e-11` |

## Timing

Existing wave2 productionish baseline:

| variant | mean ms |
| --- | ---: |
| full-P DQ/DK/diag | `1.0865` |
| serial two-pass P_TILE=64 | `1.6625` |
| P_TILE atomic+zero | `2.1496` |

Wave3 split streaming productionish before cutoff:

| variant | mean ms |
| --- | ---: |
| stream DK | `1.2461` |
| stream DQ | `1.0982` |
| tiny `R x R` blockdiag kernel | `3.0877` |
| stream total | `5.3818` |

The tiny blockdiag shape is pathological because it launched `owners*chunk =
524,288` programs. I folded blockdiag into the DK/DQ owner kernels after that,
but cutoff arrived before rerunning productionish. Folded smoke passed:

| shape | stream DK ms | stream DQ ms | total ms |
| --- | ---: | ---: | ---: |
| smoke_p128 folded | `0.0597` | `0.0527` | `0.0986` |

Smoke got slower after folding versus the earlier split smoke (`0.0627 ms`),
so the fold is not enough evidence to justify another production run.

## Blocker / Decision

No output mathematically forces materializing full `[F,F]`:

- `dqk_from_diag` only needs `chunk` many `R x R` same-token blocks.
- DSSDA is a reduction over `(i,j)` pairs and can stream tiles.
- `dk_intrachunk @ Q` and `dk_intrachunk.T @ K` can stream tiles directly.

The performance blocker is different: production weights are pair-dependent
(`causal/seg_exp`), so the useful associative reduction to `[P,N]` does not
apply. A streaming implementation must still visit the triangular `(i,j)`
surface and, unless it keeps both DK and DQ live in one large owner, recomputes
`M` tiles for DK and DQ separately. That made the measured productionish
prototype `5.3818 ms`, far behind the wave2 full-P `1.0865 ms`.

Best remaining path: not this lane as a standalone replacement. Surface
reduction is algebraically possible, but the weighted intrachunk term blocks the
cheap fusion. It is only worth revisiting if integrated into an existing owner
that already has `M` tiles live, or if the segment weight can be made separable
for the relevant term.

## Modal Cleanup

Wave3 apps launched here:

- `ap-lFmJoLjhBwKEd72i93SN61` - stopped.
- `ap-OEoFsQuPNjUw1AaJ9oYiUi` - stopped.
- `ap-HMuqTuJjKbySPXq4UCZa2v` - stopped.
- `ap-3cpdfQBf3PVQhnEDwm7bWi` - stopped.
- `ap-eN0VVof71ZGlEF9e7VK5dK` - stopped.
- `ap-0PNNXDU48UHWeL5RM2C5g6` - stopped.

Final `modal app list --json` showed all `cppmega-mamba3-bwd-bwd-surface-reduction-wave3`
apps stopped. I did not stop unrelated detached apps from other lanes.
