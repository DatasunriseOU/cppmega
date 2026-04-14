# Issue: `LayoutInference`: FloorMod const-fold divide-by-zero when modulus is dynamic inside `T.Parallel`

**Target repo:** `tile-ai/tilelang`

## Summary

`tilelang.transform.LayoutInference` crashes with
`tvm.error.InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero`
during `TryConstFold<tvm::tir::FloorMod>` when a `T.Parallel(...)` loop
body contains `csr % R` / `csr // R` indexing where `R` is a
Python-int constant closed over by an outer `@tilelang.jit`-decorated
function.

The crash fires inside the iter-map normalization that TileLang performs
while building the Fragment for the parallel-loop output buffer:

```
tvm::tl::LayoutInferencer::Substitute
  → BufferUseDefCollector::Run
  → ParallelOpNode::InferLayout
  → ParallelOpNode::CompleteBufferFragment
  → tvm::tl::infer_fragment_index
  → tvm::tl::MakeFlattenedExpression
  → tvm::arith::NormalizeIterMapToExpr
  → IterMapToExprNormalizer::ConvertIterSplitExpr
  → tvm::floormod(PrimExpr, PrimExpr, Span)
  → tvm::arith::TryConstFold<tvm::tir::FloorMod>
  → LogFatal: Divide by zero
```

`TryConstFold<FloorMod>` asserts `pb->value != 0` on the divisor; at
the point of the call the divisor has constant-folded to `0` even
though the real Python value is `R = 4`. The denominator is a
transient intermediate in the iter-map normalization — downstream
substitution would resolve it to the true constant — so the pass
should defer rather than abort.

## Reproducer

Self-contained reproducer at
[`examples/13_tilelang_floormod_dbz/`](examples/13_tilelang_floormod_dbz/):

```bash
cd examples/13_tilelang_floormod_dbz
pip install -r requirements.txt
python reproducer.py
# → prints TILELANG_BUG_REPRODUCED with full C++ backtrace
```

The reproducer drives the production `mamba_mimo_bwd_bwd` kernel from
`state-spaces/mamba` after:
1. Applying an included patch that flattens rank-3 smem operands to 2D
   (so the unrelated `LowerBulkCopy InputDim==2` path — see issue #08 /
   TileLang PR #746 — doesn't trigger first).
2. Flipping `TL_DISABLE_TMA_LOWER: True → False` and
   `TL_DISABLE_WARP_SPECIALIZED: True → False` in the
   `@tilelang.jit` PassConfig.

A CUDA device is **not** required; `LayoutInference` runs entirely on
host before codegen.

Verified on bench3 (H200 SXM, sm_90a) against `tilelang 0.1.8` built
at upstream `main` commit `f309d814` (current as of 2026-04-14).

## Trigger pattern

Inside `mamba_mimo_bwd_bwd_kernel` (state-spaces/mamba), after the
3D→2D smem flatten, the kernel contains:

```python
fused_chunk_size = chunk_size * R   # Python int, R=4

for csr, n in T.Parallel(fused_chunk_size, N):
    q_frag[csr, n] += q_bias_frag[csr % R, n]
    k_frag[csr, n] += k_bias_frag[csr % R, n]

# …

for csr, p in T.Parallel(fused_chunk_size, P):
    cs = csr // R
    r_in = csr % R
    for r_out in T.serial(R):
        csr_out = cs * R + r_out
        dPsiV_D_fused_frag[csr, p] += (
            dPhiO_shared[csr_out, p]
            * qk_dot_frag[cs, r_out * R + r_in]
            * gamma_dPsiV_frag[cs]
        )
```

All of `R`, `fused_chunk_size`, `N`, `P`, `chunk_size` are compile-time
Python ints. With `TL_DISABLE_TMA_LOWER=True` (current default) the
kernel compiles. With `TL_DISABLE_TMA_LOWER=False` (the path users hit
when enabling TMA pipelining for Hopper), `LayoutInference` fires
`TryConstFold<FloorMod>` on an intermediate with divisor `0` and
aborts.

## Error

```
tvm.error.InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero
```

Full C++ backtrace (captured via the reproducer — see README for the
complete chain):

```
tvm::tl::LayoutInferencer::Substitute
tvm::tl::BufferUseDefCollector::Run
tvm::tl::BufferUseDefCollector::FinishInferQueue
tvm::tl::BufferUseDefCollector::RunInferStep
tvm::tl::ParallelOpNode::InferLayout
tvm::tl::ParallelOpNode::CompleteBufferFragment
tvm::tl::Fragment::Fragment
tvm::tl::FragmentNode::FragmentNode
tvm::tl::infer_fragment_index
tvm::tl::MakeFlattenedExpression
tvm::arith::NormalizeIterMapToExpr
tvm::arith::IterMapToExprNormalizer::VisitExpr
tvm::arith::IterMapToExprNormalizer::ConvertIterSumExpr
tvm::arith::IterMapToExprNormalizer::ConvertIterSplitExpr
tvm::floormod
tvm::arith::TryConstFold<tvm::tir::FloorMod>
tvm::runtime::detail::LogFatal::Entry::Finalize
```

Origin: `src/arith/const_fold.h` (`TryConstFold<FloorMod>`) +
`src/arith/iter_affine_map.cc` (`IterMapToExprNormalizer::
ConvertIterSplitExpr` calling `floormod`).

## Why this matters

Mamba3 MIMO backward kernels (state-spaces/mamba) are the last mile for
NVIDIA Hopper perf on the Mamba family. Enabling TMA pipelining
(`TL_DISABLE_TMA_LOWER: False`) is the standard way to get the 20-30%
Hopper throughput win quoted by NVIDIA on similar linear-attention
kernels. With the LayoutInference crash, that path is blocked: callers
must choose between

- TMA lower OFF on bwd (current cppmega/bench3 production — throughput
  left on the table), or
- Rewriting every `% R` / `// R` site in the kernel to manually
  unrolled `T.constexpr` forms (1-2 days of kernel surgery per kernel
  family).

Similar kernels in the Mamba ecosystem (SSD, Mamba3 SIMO, Mamba3 MIMO
varlen) use the same `csr % R` / `csr // R` idiom and are reachable by
the same crash once users flip TMA on.

## Possible fixes

1. **Defer FloorMod const-fold on zero modulus.** In
   `TryConstFold<FloorMod>` (`src/arith/const_fold.h`) return
   `NullOpt` when `pb->value == 0` instead of aborting. The divisor
   here is a transient zero arising from iter-map splitting — it
   stabilizes to the real constant on the next normalization pass.

2. **Guard the `tvm::floormod` builder.** In
   `IterMapToExprNormalizer::ConvertIterSplitExpr`
   (`src/arith/iter_affine_map.cc`) preserve the symbolic `FloorMod`
   expression when the analyzer hasn't yet pinned the divisor to a
   concrete non-zero PrimExpr, instead of forcing const-fold on
   partially-substituted operands.

3. **Fail softer in `LayoutInference`.** `ParallelOpNode::InferLayout`
   could detect when the fragment index contains a `FloorMod` whose
   divisor is symbolic-but-currently-zero and fall back to a
   non-fragment layout (warn + use shared memory instead of register
   fragments) rather than hard-asserting out of the pass.

Option 1 is the least invasive and most likely localised fix.
Option 2 is the architecturally correct one (matches how TIR treats
`FloorMod` elsewhere in the simplifier). Option 3 is a safety net
independent of the arith fix.

## Workaround we're using

We keep `TL_DISABLE_TMA_LOWER: True` / `TL_DISABLE_WARP_SPECIALIZED:
True` on every bwd kernel in state-spaces/mamba's tilelang tree. This
costs ~20% end-to-end throughput at NAM56R on H200 (compared to the
TMA-on projection) but keeps compilation alive.

The obvious algebraic workaround (`csr - (csr // R) * R` in place of
`csr % R`) does **not** work: `RewriteSimplifier` / the
analyser canonicalize the subtraction form back to `FloorMod` before
`LayoutInference` runs. Empirically tested against TileLang 0.1.8 main.

Related cppmega notes:
- `reference_tma_layout_fix_broken_h200.md` — authoritative internal
  description of this specific crash.
- `reference_p1_blocked_tilelang_tma_layout.md` — how this blocks the
  Mamba3 MIMO P1 optimization.
- `docs/findings_2026_04_14_session.md` § 4 — session notes and
  upstream PR triage (PR #1458 fixes a different FloorMod site in
  the Z3 prover; PR #746 handles the sibling `InputDim==2` issue
  with warn+fallback; this bug is untouched by both).

## Environment

- PyTorch 2.12 nightly + cu132
- TileLang 0.1.8 (upstream `main` @ `f309d814`, 2026-04-14)
- TVM 0.22 (TileLang-bundled)
- tvm-ffi `<0.1.10`
- state-spaces/mamba @ 2026-04 main
- NVIDIA H200 SXM (sm_90a). Previously also seen / masked on GB10
  (sm_121a) — GB10's smem cap (99 KiB) prevented the bwd_bwd
  LayoutInference path from running, so prior "correctness passed"
  claims on GB10 were false positives for this bug specifically.
