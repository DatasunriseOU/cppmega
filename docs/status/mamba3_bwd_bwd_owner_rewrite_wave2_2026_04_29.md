# Mamba3 bwd_bwd Owner/Layout Rewrite Wave 2 - 2026-04-29

Status: evidence
Canonical: none
Date: 2026-04-29
Scope: H200 P=128 two-serial-P_TILE owner microbench for the DQ/DK/diag subproblem.

Branch: `worker/mamba3-bwd-bwd-owner-rewrite`

Base: `decad7a`

## Goal

Wave 1 proved direct p-tile atomics were correctness-feasible but too slow:
productionish isolated DQ/DK full-P was `0.8119 ms`, while atomic+zero was
`1.9991 ms`.

Wave 2 tested the next proposed ownership shape: keep one program per
`(B,H,chunk)` owner for `P=128`, run two serial `P_TILE=64` passes inside that
program, and accumulate DQ/DK/diag-like reductions on-chip. No atomics and no
global fp32 partial handoff are used by the serial2 candidate.

## Files

- Harness: `scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave2.py`
- Status: `docs/status/mamba3_bwd_bwd_owner_rewrite_wave2_2026_04_29.md`
- Comparator reused: `scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave1.py`

The wave2 harness is a Triton microbench, not a production TileLang patch. It
compares three variants on identical inputs:

- `fullp`: one owner program computes full-P DQ/DK and diag-like reductions.
- `ptile_atomic`: wave1-style two `P_TILE=64` owner programs reduce with
  `tl.atomic_add` after zeroing outputs.
- `serial2`: one owner program runs exactly two serial `P_TILE=64` passes,
  accumulates DQ/DK and two F-by-F diag-like reductions on-chip, and stores
  final outputs once.

The diag-like reductions are deliberately simple row reductions over
`qk_dot * (dphi @ psi.T)` and `qk_dot * (psi @ dphi.T)`. This keeps the
microbench focused on the cross-P live-set and accumulator question rather than
the full Mamba epilogue.

## Validation Environment

Modal H200 image:

- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- GPU: `NVIDIA H200`, capability `(9, 0)`, device count `2`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`
- Triton: `3.7.0`

Local:

```text
python -m py_compile scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave2.py
```

## H200 Runs

Smoke:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave2.py \
  --shape-csv smoke_p128 --warmup 1 --iters 3
```

App: `ap-9iLpcaETZxSEBx32gK0JVJ`, stopped normally.

Productionish:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1500 modal run \
  scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave2.py \
  --shape-csv representative,productionish --warmup 2 --iters 8
```

App: `ap-EfBs6w88EG9VESwebwGlnS`, stopped normally.

Wave1 control rerun:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave1.py \
  --shape-csv productionish --p-tile-csv 64 --warmup 2 --iters 8
```

App: `ap-Q3dXdOK4rZcvtmWyZmsLlg`, stopped normally.

## Correctness

All wave2 variants passed against full-P at `rtol=1e-2, atol=1e-2`.

Productionish max absolute diffs:

| comparison | dk | dq | diag_qk | diag_intra |
| --- | ---: | ---: | ---: | ---: |
| atomic vs full-P | `1.40e-9` | `1.40e-9` | `1.16e-10` | `8.73e-11` |
| serial2 vs full-P | `0` | `0` | `0` | `0` |

Wave1 control also passed for DQ/DK with max abs `1.40e-9`.

## Timing

Wave2 DQ/DK/diag subproblem:

| shape | programs full/serial | programs atomic | full-P ms | atomic+zero ms | serial2 ms | serial2 vs full-P | serial2 vs atomic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_p128 | 64 | 128 | 0.0406 | 0.0492 | 0.0272 | 1.49x faster | 1.81x faster |
| representative | 1,024 | 2,048 | 0.0681 | 0.1187 | 0.0834 | 1.22x slower | 1.42x faster |
| productionish | 32,768 | 65,536 | 1.0865 | 2.1496 | 1.6625 | 1.53x slower | 1.29x faster |

Wave1 DQ/DK-only control on the same image:

| shape | full-P ms | atomic compute ms | atomic+zero ms |
| --- | ---: | ---: | ---: |
| productionish | 0.8205 | 1.7973 | 2.0135 |

## Memory Model

For wave2 productionish `B=4,S=4096,H=32,N=64,P=128,R=4,chunk=16`:

| metric | full-P | serial2 | ptile_atomic |
| --- | ---: | ---: | ---: |
| program count | 32,768 | 32,768 | 65,536 |
| live input bytes per program | 73,728 | 40,960 peak/pass | 40,960 |
| accumulator bytes per program | 66,048 | 66,048 | 66,048 |
| estimated peak live bytes/program | 139,776 | 107,008 | 107,008 |
| global traffic estimate | 3,506,438,144 B | 3,506,438,144 B | 7,868,514,304 B |
| final output bytes | 1,090,519,040 B | 1,090,519,040 B | 1,090,519,040 B |
| fp32 partial handoff if written | 2,181,038,080 B | 2,181,038,080 B | n/a |

Read:

- Serial2 reduces peak live input by `1.8x` versus full-P once the qk/diag
  input is included.
- Accumulators dominate the peak. Four fp32 accumulator tiles
  (`DK`, `DQ`, `dphi@psi.T`, `psi@dphi.T`) cost `66,048` bytes/program, so
  input tiling only reduces estimated total live bytes by about `1.31x`.
- Serial2 avoids the `2.18 GiB` fp32 partial handoff and keeps the same program
  count and estimated global traffic as full-P.
- Atomic p-tiling doubles program count and has about `2.24x` the global
  traffic estimate after zeroing and atomic read-modify-write traffic.

## Decision

The two-serial-ptile owner is a real improvement over wave1 atomics, but it
does not beat full-P at production fanout. Productionish serial2 is `1.6625 ms`
versus `1.0865 ms` for the full-P DQ/DK/diag reference, even though it reduces
peak input bytes. This means H200 throughput is limited by the extra K=64 dot
passes and accumulator pressure, not by the live input bytes alone.

No full-chain integration was done because the mandatory productionish
microbench did not win. A TileLang companion kernel would likely repeat the
wave1 cross-P outcome: correct but slower unless it removes a larger source of
register/smem pressure than this isolated subproblem shows.

Wave3 should pivot, not integrate this serial2 design. The next useful lane is
to reduce or fuse the accumulator surface itself, for example by streaming only
the diag/F-by-F epilogue or splitting final epilogue work after keeping the
existing full-P DQ/DK owner. Do not spend wave3 porting serial2 P_TILE=64
ownership into TileLang as-is.

## Modal Cleanup

Apps launched by this wave:

- `ap-9iLpcaETZxSEBx32gK0JVJ` - stopped.
- `ap-EfBs6w88EG9VESwebwGlnS` - stopped.
- `ap-Q3dXdOK4rZcvtmWyZmsLlg` - stopped.

Final `modal app list --json` showed these wave apps stopped. It also showed
unrelated running apps from other lanes, including
`cppmega-tilelang-dstates-ptile-copy-probe`
`ap-2oU8mhmbIIIudNjGLlzfg3` and
`cppmega-mamba3-stage2-force-nontma-benchmark`
`ap-AbQrQ5ZwvMJfFK3E968ps4`, plus a deployed `cppmega-prebuilt-smoke` app with
zero tasks. I did not stop those because this worktree is shared with parallel
agents and this wave did not launch them.
