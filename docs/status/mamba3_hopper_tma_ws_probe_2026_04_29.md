# Mamba3 Hopper TMA/WS Probe - 2026-04-29

Branch: `worker/mamba3-hopper-tma-ws-fix`

Base: `worker/mamba3-psiv-modal-dryrun` (`ed31598`)

Goal: find a minimal temp-only source path that compiles and smoke-runs
Mamba3 MIMO `bwd_fwd` + `bwd_bwd` with Hopper TMA lowering and warp-specialized
pass config enabled.

## Context

The previous dry-run established:

- the 3D-to-2D TMA layout patch gets past the older
  `InputDim() == 2` TMA descriptor blocker;
- the patched `bwd_bwd` then hits TileLang `FloorMod` divide-by-zero;
- removing `% R` / `// R` from the relevant `T.Parallel` loops bypasses the
  DBZ but exposes `Loop layout is not injective` on
  `qk_dot_frag` / `qk_dot_shared`.

The Modal image used here reports TileLang
`0.1.8+cu132.gitf309d814`, matching the newer local TileLang generation that
already contains the post-PR-2002 pipeline/LayoutInference work. The Modal image
ref was `ghcr.io/jewelmusicee/cppmega:latest`; at run time that tag may have
pointed at an older cppmega image commit, but this probe overlays cppmega source
from the worktree.

Relevant TileLang upstream references:

- https://github.com/tile-ai/tilelang/issues/1374
- https://github.com/tile-ai/tilelang/issues/1648
- https://github.com/tile-ai/tilelang/pull/1458
- https://github.com/tile-ai/tilelang/pull/2002

## Probe Harness

Added `scripts/modal_mamba3_hopper_tma_probe.py`.

The harness:

- overlays `/home/dave/state-spaces-mamba/mamba_ssm` into the Modal container;
- copies `mamba3_mimo_bwd.py` to a temp directory;
- applies `mamba3_bwd_layout_fix.patch`;
- applies temp-only rewrites per variant;
- compiles `mamba_mimo_bwd_fwd` and `mamba_mimo_bwd_bwd`;
- smoke-runs both kernels if compile succeeds;
- records device and TileLang version metadata.

## Variants

| Variant | Result | Meaning |
| --- | --- | --- |
| `layout_patch` | fails | Original 3D-to-2D patch still hits `FloorMod` DBZ. |
| `no_floormod` | fails | `% R` removed, then `Loop layout is not injective` on `qk_dot_frag <- qk_dot_shared`. |
| `qk_serial_p` | fails | Serializing `p` avoids the exact copy error but hits `contains inner var p`. |
| `qk_shared_direct` | passes | Removes the `qk_dot_shared -> qk_dot_frag` copy and reads `qk_dot_shared` directly in dPsiV. |

The useful minimal workaround is `qk_shared_direct`:

1. keep Q/K and QK_DOT flattened to rank-2 TMA-legal views;
2. rewrite the bias and dPsiV loops so the introduced `% R` sites are gone;
3. do not allocate/copy `qk_dot_frag` for the cached diagonal qk_dot path;
4. read `qk_dot_shared[cs, r_out * R + r_in]` directly.

Materialized non-production patch artifact:

- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_hopper_tma_ws_fix.patch`

The original `mamba3_bwd_layout_fix.patch` remains unchanged as the old DBZ
reproducer input.

## Modal Results

| GPU request | Modal app | Actual device | TileLang | Result |
| --- | --- | --- | --- | --- |
| `H100:2` | `ap-IbNkH6R2N14KMvETXa7HIZ` | `NVIDIA H100 80GB HBM3` | `0.1.8+cu132.gitf309d814` | `qk_shared_direct` compile + smoke OK |
| `H200:2` | `ap-maiSlfVfhb6HD6Tui2XIWJ` | `NVIDIA H200` | `0.1.8+cu132.gitf309d814` | `qk_shared_direct` compile + smoke OK |

H100 `qk_shared_direct`:

- `bwd_fwd_source_chars=39628`
- `bwd_bwd_source_chars=89206`
- smoke: `qk_dot_absmax=0.005767822265625`
- smoke: `dq_absmax=6.148184183984995e-10`
- smoke: `dk_absmax=1.2951204553246498e-09`
- smoke: `dv_absmax=3.448803909122944e-09`

H200 `qk_shared_direct`:

- `bwd_fwd_source_chars=39628`
- `bwd_bwd_source_chars=89206`
- smoke: `qk_dot_absmax=0.005767822265625`
- smoke: `dq_absmax=6.148184183984995e-10`
- smoke: `dk_absmax=1.2951204553246498e-09`
- smoke: `dv_absmax=3.448803909122944e-09`

Both runs logged:

```text
[WS] skipped: no TMA copies in pipeline loop
```

So this validates the Hopper TMA-lowered compile path with
`TL_DISABLE_WARP_SPECIALIZED=False`, but the producer-consumer WS pass does not
actually transform these Mamba3 loops because it sees no TMA copies inside a
pipeline loop.

## Current Conclusion

The minimal Hopper compile/smoke fix is not a TileLang version bump. It is a
source-shape/layout workaround in Mamba3 `bwd_bwd`: keep cached qk_dot in shared
memory after the rank-2 TMA load and avoid the local-fragment copy that forces
LayoutInference to build a non-injective layout involving `r_out`.

This is not wired into production defaults. Next step before any production
gate is correctness comparison against the non-TMA baseline over representative
shapes, then performance measurement to decide whether keeping the shared read
is acceptable.
