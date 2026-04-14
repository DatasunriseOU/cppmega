# Regression test: `LowerBulkCopy` 3D smem fallback (guards PR #746)

**Target repo:** `tile-ai/tilelang`
**Status:** Internal regression guard — NOT an upstream issue to file.

## Summary

This template used to describe a bug to file upstream:
`LowerBulkCopy` hard-asserted `InputDim()==2`, rejecting rank-3+ shared
memory descriptors and blocking TMA on Mamba3 MIMO backward kernels.

**That has already been fixed upstream.** TileLang PR #746
([Refactor] Merge bulk copy into copy and improve layout inference for
bulk copy, merged 2025-08-21, commit `5c11d245` on
`tile-ai/tilelang:main`) replaced the hard `ICHECK(InputDim()==2)` in
`src/transform/lower_tile_op.cc` with a `LOG(WARNING)` + fallback to
`LowerNormalCopy`. Rank-3+ smem layouts now compile: TileLang prints a
warning and emits a non-bulk `cp.async` instead of crashing.

We therefore repurpose this slot as an **internal regression guard**.
If anyone later reintroduces the hard assert in `LowerBulkCopy` (or
otherwise breaks the rank-3+ fallback), our CI reproducer catches it.

## Reproducer

`examples/08_tilelang_tma_bulk_copy_3d_smem/` — three configurations:

- 3D smem `[16, 4, 64]` bfloat16 via TMA enabled path
- 4D smem variant
- Mamba3 MIMO-style rank-3 `qk_dot_shared` layout

**Validated 2026-04-14** on:

- bench3 (H200 SXM, cu13.2, TileLang `main` post-PR-#746) — 3/3 cases OK, exit 0
- GB10 (sm_121a, cu13.2, TileLang `main` post-PR-#746) — 3/3 cases OK, exit 0

## Upstream references

- **PR #746** *(merged 2025-08-21)* —
  https://github.com/tile-ai/tilelang/pull/746 — the fix we are
  regression-testing against.
- **PR #761** *(merged 2025-08-26)* —
  https://github.com/tile-ai/tilelang/pull/761 — "Add 1D TMA support",
  follow-up extending the refactor.
- **PR #2005** *(merged 2026-04-01)* —
  https://github.com/tile-ai/tilelang/pull/2005 — upstream's own 1D TMA
  regression test; our 3D/4D reproducer is the rank-3+ analogue.

## Why keep the reproducer

Mamba3 MIMO backward kernels use three rank-3 smem descriptors
(`qk_dot_shared` is structurally `[chunk_size, R, R]`; Q/K loads land
in `[chunk_size, R, N]`). These kernels depend on the PR-#746
warn-and-fallback semantics to compile at all under
`TL_DISABLE_TMA_LOWER=False`. A silent revert or re-assertion of the
2D-only check upstream would immediately break our Mamba3 MIMO bwd
build, but the failure would surface as a generic TVM InternalError
deep in a kernel log rather than as a clean test signal. Keeping the
three-case reproducer in `examples/08_…` gives us an unambiguous CI
tripwire for that regression class.

## Environment (validation run 2026-04-14)

- PyTorch 2.12 nightly + cu132
- TileLang `main` post-PR-#746 (commit with `5c11d245` in history)
- TVM 0.22 (bundled)
- NVIDIA H200 SXM (sm_90a, bench3) and NVIDIA GB10 (sm_121a)
