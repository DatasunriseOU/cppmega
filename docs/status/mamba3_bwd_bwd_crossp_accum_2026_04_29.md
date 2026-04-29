# Mamba3 bwd_bwd P_TILE Cross-P Accumulator Prototype - 2026-04-29

Branch: `worker/mamba3-bwd-bwd-crossp-accum`

Base: `worker/mamba3-stage2-force-nontma` / `972608d`

Goal: turn the prior compile-only `P_TILE=64` probe into a correctness-capable
TileLang prototype by accumulating all full-P reductions before writing bwd_bwd
outputs.

## Files

- Patch:
  `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_bwd_crossp_accum_prototype.patch`
- Harness:
  `scripts/modal_mamba3_bwd_bwd_crossp_accum.py`
- Status:
  `docs/status/mamba3_bwd_bwd_crossp_accum_2026_04_29.md`

The cross-P patch is meant to apply after
`mamba3_bwd_stage2_force_nontma.patch`.

## Design

The prototype adds `mamba_mimo_bwd_bwd_ptile_crossp_accum(...)`.

Key structural changes vs the prior compile-only probe:

- Reverse chunk loop stays outermost.
- Each chunk loops over `p_block in P/P_TILE`.
- Loop-carried `dstates` are stored in caller-provided global scratch
  `DSTATES_PTILE[B,H,n_p_tiles,N,P_TILE]`, so the kernel does not hold full
  `[N,P]` state on-chip.
- Per-chunk full-output reductions are accumulated across all P tiles before
  writes:
  - `dqk_from_diag_acc [fused_chunk_size, fused_chunk_size]`
  - `dk_acc [fused_chunk_size, N]`
  - `dk_intrachunk_acc [fused_chunk_size, fused_chunk_size]`
  - `dq_acc [fused_chunk_size, N]`
  - scalar `DDA`, `DDA_CS`, `DDA_CS_REV`, `DSSDA`, `DFACTOR`,
    `DGAMMA_DIAG`, `DANGLES`, `DD`
- `DV` and `DMIMO_V` stay P-tiled outputs because they are per-P.

The prototype disables TMA/WS for the new bwd_bwd kernel. This keeps the first
correctness prototype deterministic and avoids reintroducing the dynamic-P TMA
layout path from the compile-only probe.

## Validation

Local:

```text
python -m py_compile scripts/modal_mamba3_bwd_bwd_crossp_accum.py

patch -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_stage2_force_nontma.patch
patch --dry-run -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_bwd_crossp_accum_prototype.patch
```

Modal H200 image:

- `ghcr.io/jewelmusicee/cppmega:785c3fd`
- GPU: `NVIDIA H200`, capability `(9, 0)`, device count `2`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

### Tiny Patch-Based Smoke

Run:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py \
  --shape-csv tiny --warmup 0 --iters 1
```

App: `ap-4W9TryG5gvNw11KGy25HQm`, stopped normally.

Shape: `B=1,S=64,H=2,G=1,N=64,P=64,R=4`.

- Correctness vs stage2 baseline: `12/12` outputs allclose at `rtol=1e-2,
  atol=1e-2`.
- Stage2 bwd_bwd: `0.1377 ms`.
- Cross-P prototype bwd_bwd: `0.1542 ms`.
- Scratch: `16 KiB`.

### P=128 Smoke

Run:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1500 modal run \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py \
  --shape-csv smoke_p128 --warmup 0 --iters 1
```

App: `ap-VpnICE8y6u341hjSiS74T5`, stopped normally.

Shape: `B=1,S=256,H=4,G=1,N=64,P=128,R=4`, `P_TILE=64`.

- Correctness vs stage2 baseline: `12/12` outputs allclose.
- Stage2 bwd_bwd: `0.3148 ms`.
- Cross-P prototype bwd_bwd: `0.4068 ms`.
- Scratch: `64 KiB`.

### Productionish

Run:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1800 modal run \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py \
  --shape-csv productionish --warmup 2 --iters 6
```

App: `ap-WdcqnoCRlWQrJ1moPlPgNN`, stopped normally.

Shape: `B=4,S=4096,H=32,G=1,N=64,P=128,R=4`, `P_TILE=64`.

- Correctness vs stage2 baseline: `12/12` outputs allclose.
- Stage2 bwd_bwd mean: `3.6930 ms`
  - samples: `3.7015, 3.6885, 3.6888, 3.6924, 3.6870, 3.6997`
- Cross-P prototype bwd_bwd mean: `4.9561 ms`
  - samples: `4.9583, 4.9513, 4.9549, 4.9517, 4.9581, 4.9621`
- Slowdown vs stage2 bwd_bwd: `1.342x`.
- Scratch: `2 MiB`.

All Modal apps launched by this work completed and stopped normally.

## Read

This proves the full-output `P_TILE=64` direction is correctness-feasible in
TileLang when the full-P reductions are explicit and `dstates` is moved to a
global scratch tile. The prototype is not a performance win yet: productionish
is about 34% slower than the stage2 baseline because it serializes the P tiles
and adds global scratch traffic for `dstates`.

The next useful work is optimization, not correctness:

- Re-enable TMA/WS only for static rank-2 copies that do not depend on
  dynamic `p_start`.
- Keep `DSTATES_PTILE` in a more cache-friendly layout or use a two-kernel
  split where `dstates` and cross-P accumulators have clearer ownership.
- Consider a specialized `P=128,P_TILE=64` path with two live dstates tiles if
  the target is H200 only and smem/register pressure permits it.

## Wave 1 - P_TILE/Layout/TMA Attempt

Branch: `worker/mamba3-ptile-layout-tma`

Base evidence target: `f77c8f0`, where productionish `P_TILE=64` was
correct but slow:

- Stage2 bwd_bwd: `3.6930 ms`
- Cross-P bwd_bwd: `4.9561 ms`
- Slowdown vs stage2: `1.342x`
- Scratch: `2 MiB`

Code outcome: no kernel semantic change was kept. The harness now exposes
`--crossp-num-stages` and records stronger source markers
(`source_sha256`, TMA load/store count, mbarrier count, launch bounds, and
WS producer guard detection). The default stays `crossp_num_stages=0`, because
that is the only correctness-capable mode found in this wave.

### H200 Validation

Image/device:

- `ghcr.io/jewelmusicee/cppmega:785c3fd`
- `NVIDIA H200`, capability `(9, 0)`, device count `2`
- Torch `2.13.0.dev20260426+cu132`, CUDA `13.2`

Smoke command:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1500 modal run \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py \
  --shape-csv smoke_p128 --warmup 0 --iters 1 \
  --p-tile 64 --crossp-num-stages 0
```

App: `ap-kHUyyHSkv0Mc2lJ3u5MwgO`, stopped normally.

| shape | P_TILE | stages | correctness | stage2 ms | cross-P ms | scratch | source markers |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `smoke_p128` | 64 | 0 | `12/12` allclose | `0.3117` | `0.4006` | `64 KiB` | cross-P `tma_load_count=0`, `producer_guard=false`, launch bounds `(256,1)` |

Productionish command:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1800 modal run \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py \
  --shape-csv productionish --warmup 2 --iters 6 \
  --p-tile 64 --crossp-num-stages 0
```

App: `ap-ZO61TVAnuzFCpIQUwZckma`, stopped normally.

| shape | P_TILE | stages | correctness | stage2 ms | cross-P ms | slowdown vs stage2 | vs f77 cross-P | scratch |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `productionish` | 64 | 0 | `12/12` allclose | `3.7137` | `4.9808` | `1.341x` | `1.005x` slower | `2 MiB` |

Productionish samples:

- Stage2: `3.7116, 3.7128, 3.7141, 3.7169, 3.7149, 3.7117`
- Cross-P: `4.9914, 4.9832, 4.9740, 4.9745, 4.9792, 4.9823`

Source markers for productionish:

- Stage2 bwd_bwd: `tma_load_count=0`, `tma_store_count=0`,
  `mbarrier_wait_count=0`, `producer_guard=false`, launch bounds `(256,1)`,
  source hash `a0d4658c56a0717080a59df2ecc9a5e75f25980683b081b9b7606682eb7166cf`.
- Cross-P bwd_bwd: `tma_load_count=0`, `tma_store_count=0`,
  `mbarrier_wait_count=0`, `producer_guard=false`, launch bounds `(256,1)`,
  source hash `6101d21f2bd78d644d209d23a676dc40f226a12675926c568a8b466114455a18`.

### Blocked Variants

No performance win was found before cutoff.

| variant | app | result |
| --- | --- | --- |
| TMA+WS pass_configs on cross-P | `ap-ec0qSMP2jCagbyO66xu7cs`, `ap-ANV25CfKNjO9P3079VxSgE` | Remote worker exited before SUMMARY_JSON; tiny isolated the crash to import of the patched module after setting cross-P `TL_DISABLE_WARP_SPECIALIZED=False`. |
| TMA-only plus scratch TMA guards | `ap-tvtnvIKMzQzJYAH6hqu5Z4` | Import succeeded, but cross-P compile failed: `target function must be a PrimFunc but got <class 'NoneType'>`. |
| Scratch `DSTATES_PTILE` per-copy `disable_tma=True` with safe pass configs | `ap-wTxLNiUTF6rdKHVjmyPhqT` | `P_TILE=64` smoke compile failed with the same `PrimFunc None` error; the guard edit was reverted. |
| `P_TILE=32`/`128` under the scratch-guard variant | `ap-bWKS4E3X869v73kU8l5f3Z`, `ap-Gtpi3UYf4GnEQV6MZt5HjY` | Compile failed with `PrimFunc None`. This is not a clean verdict on P_TILE alone because the scratch guard was present. |
| `crossp_num_stages=1`, TMA/WS disabled | `ap-TrKGIP5GWwx9dFPM1CAJCd` | Compile failed in pipeline planning: overlapping writes to `q_shared` in stages 8 and 11. |

Modal cleanup: after cutoff, `modal app list` showed all
`cppmega-mamba3-bwd-bwd-crossp-accum` apps stopped with `0` tasks. One unrelated
deployed `cppmega-pre...` app was left untouched.

### Read For Next Wave

Wave 1 did not return performance. The current cross-P design remains
correctness-capable only in the f77 mode: `P_TILE=64`, `num_stages=0`,
TMA/WS disabled. The main blocker is not just throughput; several seemingly
small TileLang changes fail before source generation.

Recommended next wave:

1. Do not start by enabling WS on the full cross-P kernel. First isolate a
   minimal kernel containing the `DSTATES_PTILE` scratch copy and prove which
   `T.copy(..., disable_tma=...)` forms lower to a PrimFunc.
2. After that, retest `P_TILE=32/128` without the scratch-guard confounder.
3. If P_TILE variants still compile only at 64, focus on reducing global scratch
   traffic or splitting state passing from full-P reductions rather than trying
   `num_stages=1` on the current monolithic loop.

## Wave 2 - DSTATES_PTILE Minimal Scratch-Copy Legality

Status: `evidence`

Branch: `worker/mamba3-ptile-layout-tma`

Base: `d543a4a`

Cutoff: stopped launching new experiments at user request after the minimal
scratch-copy probe and one clean full-kernel `P_TILE=32` smoke. The full
`P_TILE=32/64/128` matrix did not complete, so there is no clean productionish
table for this wave.

### Added Reproducer

- `scripts/modal_tilelang_dstates_ptile_copy_probe.py`

The script is a self-contained Modal H200 harness for four minimal copy forms:

- `rank2_dynamic_fragment`: global rank-2 `[B*H*N, P]` scratch with dynamic
  `p_start = p_block * P_TILE`, copied directly to a fragment and back.
- `rank2_dynamic_shared`: the same rank-2 dynamic `p_start` descriptor copied
  global -> shared -> fragment -> shared -> global.
- `rank5_static_fragment`: DSTATES-like rank-5 `[B,H,n_p_tiles,N,P_TILE]`
  descriptor copied directly to a fragment and back.
- `rank5_static_shared`: the same rank-5 descriptor copied global -> shared ->
  fragment -> shared -> global.

Each form is compiled/run with:

- TMA lowering enabled and disabled.
- Per-copy `disable_tma=False` and `disable_tma=True`.
- Shape `B=1,H=2,N=16,P=128,P_TILE=64`, dtype `bfloat16`.

Local validation:

```text
python -m py_compile scripts/modal_tilelang_dstates_ptile_copy_probe.py
```

### Minimal H200 Scratch-Copy Probe

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:1 timeout 1200 modal run \
  scripts/modal_tilelang_dstates_ptile_copy_probe.py
```

App: `ap-2oU8mhmbIIIudNjGLlzfg3`, stopped normally.

Image/device:

- `ghcr.io/jewelmusicee/cppmega:785c3fd`
- `NVIDIA H200`, capability `(9, 0)`, device count `1`
- Torch `2.13.0.dev20260426+cu132`, CUDA `13.2`
- TileLang `0.1.8+cu132.gitf309d814`

Result: all 16 combinations compiled and ran a roundtrip smoke with
`max_abs=0.0` and exact `allclose_0=true`.

| form | TMA lower | per-copy `disable_tma` | compile | run | TMA source markers |
| --- | --- | --- | --- | --- | --- |
| `rank2_dynamic_fragment` | on/off | false/true | ok | exact roundtrip | no TMA load/store |
| `rank2_dynamic_shared` | on | false | ok | exact roundtrip | `tma_store_count=3`, no TMA load |
| `rank2_dynamic_shared` | on | true | ok | exact roundtrip | no TMA load/store |
| `rank2_dynamic_shared` | off | false/true | ok | exact roundtrip | no TMA load/store |
| `rank5_static_fragment` | on/off | false/true | ok | exact roundtrip | no TMA load/store |
| `rank5_static_shared` | on | false | ok | exact roundtrip | `tma_store_count=3`, no TMA load |
| `rank5_static_shared` | on | true | ok | exact roundtrip | no TMA load/store |
| `rank5_static_shared` | off | false/true | ok | exact roundtrip | no TMA load/store |

Legal minimal forms:

- Rank-2 flattened global scratch with dynamic `p_start` indexing is legal for
  both fragment-direct and shared-staged copies.
- Rank-5 DSTATES-like scratch is legal for both fragment-direct and
  shared-staged copies.
- Per-copy `disable_tma=True` is legal in the minimal reproducer and suppresses
  the TMA store lowering on shared-staged global stores.

Illegal minimal forms:

- None found in the isolated reproducer. The Wave 1
  `target function must be a PrimFunc but got <class 'NoneType'>` blocker does
  not reproduce from scratch-copy legality alone.

### Full Cross-P Retest

No scratch-guard edits were applied. The full harness used the existing
`mamba3_bwd_stage2_force_nontma.patch` plus
`mamba3_bwd_bwd_crossp_accum_prototype.patch`.

Clean smoke:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1500 modal run \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py \
  --shape-csv smoke_p128 --warmup 0 --iters 1 \
  --p-tile 32 --crossp-num-stages 0
```

App: `ap-UxkhUKErJcz4o0kdV1h3vF`, stopped normally.

| shape | P_TILE | correctness | stage2 bwd_bwd | cross-P bwd_bwd | slowdown | scratch | source markers |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `smoke_p128` | 32 | `12/12` allclose | `0.3110 ms` | `0.4236 ms` | `1.362x` | `64 KiB` | cross-P `tma_load_count=0`, `tma_store_count=0`, `dynamic_scratch_tma_guarded=false` |

Comparison points:

- Versus Wave 1 / d543 `P_TILE=64` smoke (`0.4006 ms`), this `P_TILE=32`
  smoke is about `1.057x` slower.
- The Wave 1 / d543 productionish reference remains stage2 `3.7137 ms`,
  cross-P `4.9808 ms`; Wave 2 did not reach productionish retests before
  cutoff.

Blocked/incomplete full retests:

| variant | app | result |
| --- | --- | --- |
| `P_TILE=64`, first retry | `ap-y4qGZnHG7MQuJ2DgLQBHU7` | Modal stopped before import completed; no SUMMARY_JSON and no kernel verdict. |
| `P_TILE=64`, second retry | `ap-6hC2ck7AJAIfsMzQbVM2kG` | Interrupted during TileLang/NVCC compile with `KeyboardInterrupt` and `RemoteError`; no SUMMARY_JSON and no kernel verdict. |
| `P_TILE=128` | not launched | Cutoff arrived before launch. |
| productionish `P_TILE=32/64/128` | not launched | Cutoff arrived before productionish matrix. |

Modal cleanup: all apps launched by this wave are stopped with `0` tasks. The
pre-existing deployed `cppmega-pre...` app was left untouched.

### Wave 2 Read

The minimal legality target was successful: scratch-copy forms are not the
isolated cause of `PrimFunc None`. The remaining blocker is integration-specific
in the full cross-P kernel or in the prior scratch-guard edit context.

Wave 3 recommendation:

1. Keep the new minimal scratch-copy probe as a regression harness.
2. Resume full-kernel retest from `P_TILE=64` and `P_TILE=128` without changing
   scratch guards; use longer Modal timeouts or a prewarmed compile cache to
   avoid the NVCC interruption seen here.
3. If `P_TILE=64/128` compile cleanly, run the productionish table for
   `P_TILE=32/64/128` against stage2 and d543/f77 references.
4. Do not spend Wave 3 on isolated scratch `disable_tma=True`; the standalone
   forms are legal. Focus on the full-kernel interaction around pipeline layout,
   scratch copy placement, and prior scratch-guard transformations.

## Wave 3 - Full-Kernel P_TILE Matrix Attempt

Status: `cutoff_incomplete`

Branch: `worker/mamba3-ptile-layout-tma`

Harness update:

- `scripts/modal_mamba3_bwd_bwd_crossp_accum.py` now accepts `--run-id`.
- Each remote run writes `report.json` to Modal volume
  `cppmega-mamba3-benchmarks/mamba3_bwd_bwd_crossp_accum/<run_id>/`.
- The local entrypoint also has `--spawn-only` for `modal run --detach`, which
  avoided local RPC disconnects killing long TileLang/NVCC compiles.

No scratch-guard edits were applied. The full harness used
`mamba3_bwd_stage2_force_nontma.patch` plus
`mamba3_bwd_bwd_crossp_accum_prototype.patch`.

### Completed Smoke Runs

Commands used the detached/spawn path, one variant per app:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 modal run --detach \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py --spawn-only \
  --run-id wave3_smoke_p128_ptile32_dspawn1 \
  --shape-csv smoke_p128 --warmup 0 --iters 1 \
  --p-tile 32 --crossp-num-stages 0

GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 modal run --detach \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py --spawn-only \
  --run-id wave3_smoke_p128_ptile64_dspawn1 \
  --shape-csv smoke_p128 --warmup 0 --iters 1 \
  --p-tile 64 --crossp-num-stages 0
```

| shape | P_TILE | app | status | correctness | stage2 bwd_bwd | cross-P bwd_bwd | slowdown | scratch | cross-P source markers |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `smoke_p128` | 32 | `ap-1qIdttCKabrRdLADafQd5y` | ok | `12/12` allclose | `0.3186 ms` | `0.4477 ms` | `1.405x` | `64 KiB` | hash `9e3a4504169e9cb71510754b574dd7ecea7b7d9d091891209aef0cee68e5c9e0`, TMA load/store `0/0`, mbarrier `0`, launch bounds `(256,1)`, `dynamic_scratch_tma_guarded=false` |
| `smoke_p128` | 64 | `ap-YxVpvOoTTPTWpxKxBWCtuL` | ok | `12/12` allclose | `0.3399 ms` | `0.4385 ms` | `1.290x` | `64 KiB` | hash `729a7667a0902ebc5e82e84344d4a0d9b8a2ea809860a08a9384632397bab43e`, TMA load/store `0/0`, mbarrier `0`, launch bounds `(256,1)`, `dynamic_scratch_tma_guarded=false` |
| `smoke_p128` | 128 | not launched | blocked | cutoff | - | - | - | - | user cutoff arrived before launch |

Baseline source hashes:

- `P_TILE=32` run stage2: `140adea38491302a2e2516af331ca48537538b821a8c26c2472739f12af9844d`
- `P_TILE=64` run stage2: `34aa8f1e82424ce2a48d25d5827a3864e031bf7d01326952237322c1540c2630`

### Interruption Log

Before switching to `--detach --spawn-only`, two direct local RPC attempts did
not produce kernel verdicts:

| run id | app | result |
| --- | --- | --- |
| `wave3_smoke_p128_ptile32` | `ap-oSR4Ffp7FP4E3giFrDJUsI` | Modal stopped during baseline `mamba_mimo_bwd_bwd_kernel` compile with `KeyboardInterrupt` in `tilelang/tileop/gemm/__init__.py`; no `SUMMARY_JSON`, no `report.json`. |
| `wave3_smoke_p128_ptile32_r1` | `ap-8RfODp06sGEf05zSQJWt0V` | Immediate Modal `RemoteError` after object creation; app logs only `Stopping app - user stopped from CLI`; no kernel verdict. |

`modal run --detach` with `.remote()` was also rejected by Modal's own warning:
detached `.remote()` calls may be canceled when the local caller disconnects.
The working orchestration is `modal run --detach ... --spawn-only`.

### Cutoff Read

The requested full matrix is incomplete because the user cutoff arrived after
`P_TILE=64` smoke completed and before launching `P_TILE=128` or productionish.
No productionish Wave 3 runs were launched.

From the completed clean smoke rows, `P_TILE=64` is the best of the two tested
variants but still slower than stage2. There is no evidence yet that P tiling
recovers performance; however, Lane B cannot be formally declared dead until
`P_TILE=128` smoke is run, because that is the full-P ownership case.

Wave 4 recommendation:

1. Start with exactly one detached/spawn smoke run:
   `P_TILE=128`, `shape_csv=smoke_p128`, `warmup=0`, `iters=1`.
2. If it compiles and is correct, run productionish only for the best smoke row
   among `P_TILE=64` and `P_TILE=128`; skip `P_TILE=32` productionish unless a
   repeat smoke contradicts the slower result.
3. If `P_TILE=128` is also slower than stage2, declare Lane B dead for P-tiling
   and pivot away from the monolithic P-tiled cross-P accumulator. The next
   pivot should be a two-kernel split or state-passing/scratch-traffic reduction,
   not TMA/WS.

Modal cleanup: after cutoff, `modal app list` showed all
`cppmega-mamba3-bwd-bwd-crossp-accum` apps stopped with `0` tasks. The
pre-existing deployed `cppmega-pre...` app was left untouched.

## Wave 4 - Lane B Closeout

Status: `closed_dead`

Branch: `worker/mamba3-ptile-layout-tma`

No harness or kernel changes were made in Wave 4. Runs used the existing
detached/spawn harness and wrote `report.json` to Modal Volume
`cppmega-mamba3-benchmarks`.

Commands:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 modal run --detach \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py --spawn-only \
  --run-id wave4_smoke_p128_ptile128_dspawn1 \
  --shape-csv smoke_p128 --warmup 0 --iters 1 \
  --p-tile 128 --crossp-num-stages 0

GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 modal run --detach \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py --spawn-only \
  --run-id wave4_productionish_ptile64_dspawn1 \
  --shape-csv productionish --warmup 2 --iters 6 \
  --p-tile 64 --crossp-num-stages 0

GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 modal run --detach \
  scripts/modal_mamba3_bwd_bwd_crossp_accum.py --spawn-only \
  --run-id wave4_productionish_ptile128_dspawn1 \
  --shape-csv productionish --warmup 2 --iters 6 \
  --p-tile 128 --crossp-num-stages 0
```

Artifacts:

- `/benchmarks/mamba3_bwd_bwd_crossp_accum/wave4_smoke_p128_ptile128_dspawn1/report.json`
- `/benchmarks/mamba3_bwd_bwd_crossp_accum/wave4_productionish_ptile64_dspawn1/report.json`
- `/benchmarks/mamba3_bwd_bwd_crossp_accum/wave4_productionish_ptile128_dspawn1/report.json`

Smoke matrix:

| shape | P_TILE | run/app | status | correctness | stage2 bwd_bwd | cross-P bwd_bwd | slowdown | scratch | cross-P source markers |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `smoke_p128` | 32 | Wave 3 `ap-1qIdttCKabrRdLADafQd5y` | ok | `12/12` allclose | `0.3186 ms` | `0.4477 ms` | `1.405x` | `64 KiB` | TMA load/store `0/0`, mbarrier `0`, launch bounds `(256,1)`, `dynamic_scratch_tma_guarded=false` |
| `smoke_p128` | 64 | Wave 3 `ap-YxVpvOoTTPTWpxKxBWCtuL` | ok | `12/12` allclose | `0.3399 ms` | `0.4385 ms` | `1.290x` | `64 KiB` | TMA load/store `0/0`, mbarrier `0`, launch bounds `(256,1)`, `dynamic_scratch_tma_guarded=false` |
| `smoke_p128` | 128 | Wave 4 `ap-wP6xpi6k3Oq7an86mIYef2` | ok | `12/12` allclose | `0.3359 ms` | `0.3813 ms` | `1.135x` | `64 KiB` | hash `eddce278314dc802d36634bb9e140e873d217cecce35eca9efa60948dcb17d33`, TMA load/store `0/0`, mbarrier `0`, launch bounds `(256,1)`, `dynamic_scratch_tma_guarded=false` |

Productionish matrix:

| shape | P_TILE | run/app | status | correctness | stage2 bwd_bwd | cross-P bwd_bwd | slowdown | scratch | cross-P source markers |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `productionish` | 64 | `wave4_productionish_ptile64_dspawn1` / `ap-sZ5cqtHjWnZkqUQ4pnylxE` | ok | `12/12` allclose | `3.6972 ms` | `4.9464 ms` | `1.338x` | `2 MiB` | hash `fdb7188c19c3ff4cccab8ecb0db906a2d9bc38ab55cb3e2dcbb0a1cbf39f13e1`, TMA load/store `0/0`, mbarrier `0`, launch bounds `(256,1)`, `dynamic_scratch_tma_guarded=false` |
| `productionish` | 128 | `wave4_productionish_ptile128_dspawn1` / `ap-jJeiRhbxgE1aB5xsY9s4U7` | ok | `12/12` allclose | `3.7179 ms` | `4.4565 ms` | `1.199x` | `2 MiB` | hash `109c34f7e0497314b38aeca27c4f01ecb6ba2a4d3f9928c5422bc83b994e5120`, TMA load/store `0/0`, mbarrier `0`, launch bounds `(256,1)`, `dynamic_scratch_tma_guarded=false` |

`P_TILE=32` productionish was intentionally skipped: its clean smoke row was
already slower than both `P_TILE=64` and `P_TILE=128`, so it was not a cheap
candidate winner.

Lane B verdict: P-tiling is dead for the monolithic cross-P accumulator. The
best row is `P_TILE=128`, but it remains slower than stage2 on both smoke
(`1.135x`) and productionish (`1.199x`). Do not spend more cycles on P_TILE
layout/TMA/WS variants in this lane; the next viable pivot is reducing
state-passing/scratch traffic or splitting the work across kernels.

Modal cleanup: all Wave 4 apps were stopped after reports were collected. The
pre-existing deployed `cppmega-pre...` app was left untouched.
