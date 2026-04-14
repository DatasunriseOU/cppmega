# Reproducer: TileLang `LowerBulkCopy` rank-3+ shared-memory handling

Checks that TileLang compiles a kernel with a rank-3 `alloc_shared` +
`T.copy(global[...], shared[...])` without the historical hard
`ICHECK(shared_layout->InputDim() == 2)` / "Cannot detect TMA layout"
abort — now a `LOG(WARNING)` + graceful fallback to `LowerNormalCopy`
after upstream PR #746.

## Upstream history

| Date (UTC) | TileLang version | Behavior on rank-3 `T.alloc_shared` + `T.copy` |
| --- | --- | --- |
| ≤ 2025-08-21 | 0.1.7 and earlier | `tvm::tl::CopyNode::LowerBulkCopy` hard-asserts: `Check failed: (shared_layout->InputDim() == 2) is false: Cannot detect TMA layout.` Compile aborts. |
| 2025-08-22 | main @ `5c11d24` (PR #746 merged) | Assert replaced with `LOG(WARNING) "TMA bulk copy cannot support shared layout with input dimension N, fallback to normal copy."` Compile succeeds; kernel lowers through `LowerNormalCopy` (non-bulk `cp.async`). |
| current (2026-04) | 0.1.8+cuda.git `f309d81` on bench fleet | Same as PR #746 — warn+fallback. Verified on GB10 (sm_121a) via this reproducer. |

Upstream PR: <https://github.com/tile-ai/tilelang/pull/746>
(`[Refactor] Merge bulk copy into copy and improve layout inference for
bulk copy`).

Relevant source: `src/op/copy.cc`, `CopyNode::LowerBulkCopy`, around
`if (shared_layout->InputDim() < 2) { LOG(WARNING) ...; return
LowerNormalCopy(T, analyzer); }`.

## What this reproducer does

Compiles three variants of the kernel from the upstream_prs/08
template and reports status:

1. **[A] rank-3 smem, TMA enabled** — the bug path. Pre-746 aborted
   here with `InternalError`. Post-746 must compile (warn+fallback is
   fine and expected). A hard abort is treated as a regression → exit 1.
2. **[B] rank-3 smem, TMA disabled** — baseline. Should always work;
   a failure here indicates a broader breakage.
3. **[C] rank-2 smem, TMA enabled** — fast-path sanity. Rank-2 was
   always supported; failure here means the TMA path is itself broken.

Exit code `0` means all three compile and PR #746 behavior is intact.
Exit code `1` means the hard-assert is back (or some other regression
landed that prevents compilation).

## Run

```bash
pip install -r requirements.txt
python reproducer.py
```

Requires a CUDA device. TileLang lowers via NVRTC/NVCC; we ran it on:

- NVIDIA GB10 (sm_121a) — `cppmega-venv` on `gb10` host.
- NVIDIA H200 SXM (sm_90a) — bench3 `/mnt/data/venv`.

Both report status `OK` on `tilelang 0.1.8+cuda.gitf309d814`.

## What a regression looks like

If PR #746 were reverted, or a new change re-introduces the hard
`ICHECK` on `InputDim() == 2`, case [A] would output:

```
[A] rank-3 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=False
    STATUS: REGRESSION — hard-assert on 'Cannot detect TMA layout'
    ...
      tvm.error.InternalError: Check failed: (shared_layout->InputDim() == 2)
      is false: Cannot detect TMA layout.
```

and the script exits `1`. Any other uncaught exception in [A], [B],
or [C] is likewise treated as a regression.

## Expected output (post-746, current)

```
TILELANG_VERSION : 0.1.8+cuda.gitf309d814
DEVICE           : NVIDIA GB10 (sm_121)
PR #746 (LowerBulkCopy warn+fallback) merged upstream 2025-08-22.
EXPECTED         : warn-not-assert on rank-3 smem (PR #746 behavior).
REGRESSION MARKER: any 'Cannot detect TMA layout' → exit 1.

[A] rank-3 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=False
    STATUS: OK — compile succeeded (warn+fallback logged)

[B] rank-3 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=True  (baseline)
    STATUS: OK

[C] rank-2 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=False (fast-path)
    STATUS: OK — rank-2 TMA fast-path compiled cleanly
========================================================================
VERDICT: OK. PR #746 warn+fallback behavior intact on tilelang 0.1.8+cuda.gitf309d814.
         (Note: 3D smem still falls back to cp.async — TMA 3D
          descriptor support remains a separate feature request.)
```

Exit code: **0**.

## Follow-up

PR #746 removed the crash; it did **not** add native 3D TMA descriptor
emission. `cp.async.bulk.tensor.{3,4,5}d` in PTX is still unused by
TileLang. This is tracked as a separate feature request in
`cppmega/upstream_prs/08_tilelang_tma_bulk_copy_3d_smem_issue.md`
("Possible fix 2: extend `DetectTMALayout` to rank-3+").

Practical impact on NAM56R: Mamba3 MIMO backward `qk_dot_shared`
(`[chunk_size, R, R]`) and Q/K smem views (`[chunk_size, R, N]`)
currently compile but do not get the TMA+warp-spec throughput win
(~20–30% reported by NVIDIA on similar kernels). See
`reference/reference_p1_blocked_tilelang_tma_layout.md` in project
memory for the workaround status.
