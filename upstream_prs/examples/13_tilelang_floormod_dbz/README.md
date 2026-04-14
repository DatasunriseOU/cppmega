# Reproducer: `LayoutInference` FloorMod const-fold divide-by-zero

Compiles the Mamba3 MIMO `bwd_bwd` kernel (state-spaces/mamba) with TMA
lowering and warp specialization enabled. After applying the included
rank-3 → rank-2 smem flatten patch (which is how you sidestep the separate
`LowerBulkCopy InputDim==2` assert tracked in
[../../08_tilelang_tma_bulk_copy_3d_smem_issue.md](../../08_tilelang_tma_bulk_copy_3d_smem_issue.md)),
`tilelang.transform.LayoutInference` crashes during `TryConstFold<FloorMod>`
with `Check failed: pb->value != 0 (0 vs. 0) : Divide by zero`.

Verified 2026-04-14 on bench3 (H200 SXM, sm_90a) against
`tilelang 0.1.8+cuda.gitf309d814` (upstream main at commit `f309d814`).

## Run

```bash
pip install -r requirements.txt
python reproducer.py
```

The reproducer needs only `tilelang` + `mamba_ssm` importable — the
FloorMod crash fires inside the `LayoutInference` TIR pass, which runs on
host before any CUDA codegen, so a CUDA device is **not** required.

## Expected output — bug present (current TileLang main)

```
Upstream mamba3_mimo_bwd.py: /.../mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py
Patched + TMA-flipped copy:   /tmp/tilelang_floormod_repro_*/mamba3_mimo_bwd.py
Compiling mamba_mimo_bwd_bwd at B=1 S=64 H=4 G=1 N=64 P=64 R=4 (TMA+WS = ON)...
...
CRASH: InternalError
Check failed: pb->value != 0 (0 vs. 0) : Divide by zero

Traceback (most recent call last):
  ...
  File "<unknown>", line 0, in tvm::tl::LayoutInferencer::Substitute(tvm::tir::PrimFunc, bool)
  File "<unknown>", line 0, in tvm::tl::BufferUseDefCollector::Run()
  File "<unknown>", line 0, in tvm::tl::ParallelOpNode::InferLayout(tvm::tl::LayoutInferArgs const&, tvm::tl::InferLevel) const
  File "<unknown>", line 0, in tvm::tl::ParallelOpNode::CompleteBufferFragment(tvm::tir::Buffer const&) const
  File "<unknown>", line 0, in tvm::tl::infer_fragment_index(...)
  File "<unknown>", line 0, in tvm::tl::MakeFlattenedExpression(...)
  File "<unknown>", line 0, in tvm::arith::NormalizeIterMapToExpr(tvm::PrimExpr const&)
  File "<unknown>", line 0, in tvm::arith::IterMapToExprNormalizer::ConvertIterSplitExpr(...)
  File "<unknown>", line 0, in tvm::floormod(tvm::PrimExpr, tvm::PrimExpr, tvm::Span)
  File "<unknown>", line 0, in tvm::arith::TryConstFold<tvm::tir::FloorMod>(PrimExpr, PrimExpr)
  File "<unknown>", line 0, in tvm::runtime::detail::LogFatal::Entry::Finalize()
tvm.error.InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero

TILELANG_BUG_REPRODUCED: LayoutInference FloorMod divide-by-zero
```

Exit code: **1**.

## Expected output — bug fixed

```
OK: compiled cleanly (CUDA source NNNNN chars).
TILELANG_BUG_NOT_REPRODUCED
```

Exit code: **0**.

## What the reproducer does

1. Locates the installed `mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd`
   upstream source file.
2. Copies it to a temp dir and applies `mamba3_bwd_layout_fix.patch`,
   which flattens four rank-3 shared-memory operands
   (`qk_dot_shared [chunk_size, R, R]`, Q/K views, QK_DOT global signature)
   down to rank-2 shapes so the TMA bulk-copy lowerer (which requires
   `InputDim == 2`) doesn't assert.
3. Flips the `@tilelang.jit` PassConfig for that file:
   - `TL_DISABLE_TMA_LOWER`: `True` → `False`
   - `TL_DISABLE_WARP_SPECIALIZED`: `True` → `False`
4. Imports the patched+flipped module and invokes `mamba_mimo_bwd_bwd`
   at a tiny NAM56R-like shape (`B=1 S=64 H=4 G=1 N=64 P=64 R=4,
   chunk_size=16, rotary_dim_divisor=4, dtype=float16`).
5. TileLang starts compiling; `LayoutInference` runs
   `BufferUseDefCollector::Run → ParallelOpNode::InferLayout →
   CompleteBufferFragment → infer_fragment_index → MakeFlattenedExpression
   → NormalizeIterMapToExpr → IterMapToExprNormalizer::ConvertIterSplitExpr
   → tvm::floormod → TryConstFold<FloorMod>`, which asserts
   `pb->value != 0` and crashes with `Divide by zero`.

## Root cause (informal)

The Mamba3 `bwd_bwd` kernel expresses several layout transforms as
`csr % R` / `csr // R` inside `T.Parallel(fused_chunk_size, N)` and
`T.Parallel(fused_chunk_size, P)` where `fused_chunk_size = chunk_size * R`
and `R` is a Python int (= 4 in our config) closed over by the outer
`@tilelang.jit` function. During `ParallelOpNode::CompleteBufferFragment`,
TileLang calls `infer_fragment_index` → `MakeFlattenedExpression` which
normalizes the iter map to a `FloorMod(csr, d)` expression. At some
intermediate canonicalization step the denominator `d` of that `FloorMod`
is substituted (or simplified) to `0` before the real constant value (4)
is resolved, and `TryConstFold<FloorMod>` asserts on divide-by-zero
without guarding against the transient zero.

Evidence:
- Bug does not fire when TMA lower + warp-spec are OFF, even though the
  same `csr % R` loops are present. The bug is specific to the
  `LayoutInference` → iter-map normalization path that TMA pipelining
  activates.
- Bug does not fire on `bwd_fwd` in the same file (different fragment
  completion path, likely a different iter-map structure).
- Minimal standalone kernels with `csr % R` inside `T.Parallel` compile
  cleanly even with TMA lower ON — the bug needs the full bwd_bwd
  fragment/buffer graph to trip the specific `TryConstFold` site.

## Proposed fixes

Listed from least to most invasive:

1. **Guard `TryConstFold<FloorMod>`** against a zero modulus by returning
   an unfolded `FloorMod(a, b)` node when `b` constant-folds to 0 at an
   intermediate step (rather than aborting). The 0 modulus here is
   spurious — downstream substitution replaces it with the real constant
   — so the pass should defer rather than crash.
2. **Preserve `FloorMod` until concrete substitution** in
   `IterMapToExprNormalizer::ConvertIterSplitExpr`: emit the FloorMod
   node untouched when the simplifier hasn't yet resolved the divisor to
   a non-zero PrimExpr.
3. Document this limitation in TileLang and surface a pre-check during
   `ParallelOpNode::InferLayout` that rejects modulo indexing with
   symbolic divisors, falling back to non-TMA codegen instead of
   asserting.

## Workaround we're using (kernel side)

Replacing `csr % R` with the algebraic form `csr - (csr // R) * R` does
**not** work — `RewriteSimplifier` canonicalizes it back to `FloorMod`
before `LayoutInference` runs. The options that do work are:

- **Unroll R at compile time** by making it a `T.constexpr` and
  enumerating the R dimension manually with `for r in T.unroll(R):`.
  Eliminates the FloorMod entirely. Cost: 1-2 days kernel surgery,
  tracked internally under P1 follow-up.
- **Keep TMA lower OFF on bwd kernels** (the current production path on
  bench3 and europe — `TL_DISABLE_TMA_LOWER: True`). Loses the 20-30%
  Hopper TMA win but keeps the compile alive.

## Environment

- PyTorch 2.12 nightly + cu132
- TileLang 0.1.8 built at upstream main commit `f309d814`
- TVM 0.22 (TileLang-bundled)
- NVIDIA H200 SXM (sm_90a). Also verified GB10 (sm_121a) compile path
  when the kernel actually executes; see
  [../../../docs/findings_2026_04_14_session.md](../../../docs/findings_2026_04_14_session.md)
  for why the prior GB10 "correctness pass" was a false positive
  (smem cap prevented the bwd_bwd LayoutInference path from being
  exercised).

## References

- `upstream_prs/08_tilelang_tma_bulk_copy_3d_smem_issue.md` — sibling
  issue covering the `LowerBulkCopy InputDim==2` assert (separate bug,
  addressed upstream by PR #746 warn+fallback).
- `docs/findings_2026_04_14_session.md` — session writeup covering why
  the `tma-layout-fix-3d-to-2d` branch must not be merged on H200
  (csr % R inside T.Parallel trips LayoutInference FloorMod DBZ); this
  pack is the minimal repro of that blocker.
- TileLang PR [#1458](https://github.com/tile-ai/tilelang/pull/1458) —
  fixed a different FloorMod bug in the Z3 prover; does **not** address
  the const-fold DBZ path reproduced here.
