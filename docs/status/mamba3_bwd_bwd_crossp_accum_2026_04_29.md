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
