"""Reproducer: TileLang `LowerBulkCopy` rank-3+ shared-memory descriptors.

Historical behavior (TileLang ≤ 0.1.7, pre-PR #746 merged 2025-08-22):
    Compile aborts with a hard InternalError:

        tvm.error.InternalError: Check failed:
        (shared_layout->InputDim() == 2) is false: Cannot detect TMA layout.

    Origin: `tvm::tl::CopyNode::LowerBulkCopy` in `src/op/copy.cc`.
    The assertion forbade any `T.copy(global[...], shared[...])` whose
    shared destination had rank > 2, even though PTX `cp.async.bulk.tensor.{3,4,5}d`
    exists. Our Mamba3 MIMO backward kernels hit this because `qk_dot_shared`
    is structurally `[chunk_size, R, R]` and Q/K loads land in
    `[chunk_size, R, N]` smem views.

Current behavior (TileLang main, PR #746 merged):
    The hard ICHECK is replaced by a `LOG(WARNING)` + graceful fallback to
    `LowerNormalCopy` (the non-bulk `cp.async` path). Compile succeeds. The
    kernel still does not get TMA + warp-spec on the rank-3 copy, so the
    "proper fix" (extending `DetectTMALayout` to emit 3D descriptors) is
    still pending — but there is no longer a compile-time crash.

What this reproducer verifies:

  (1) The 3D-smem kernel from the upstream issue template compiles on the
      installed TileLang. Any `InternalError` / `ICHECK` / `LOG(FATAL)`
      containing "Cannot detect TMA layout" is treated as a regression of
      PR #746 → exit 1.

  (2) The same 3D-smem kernel also compiles with `TL_DISABLE_TMA_LOWER=True`
      (sanity check — this path was never affected by the assert).

  (3) For context, the 2D-smem case (rank-2 shared) compiles cleanly with
      TMA enabled — this is the fast-path the assert was originally
      protecting.

Exit codes:
    0  — all three cases compile. PR #746 behavior intact. (Post-merge: good.)
    1  — (1) hard-asserts on "Cannot detect TMA layout" OR any case fails
         with a non-fallback error. Regression / pre-746 TileLang detected.

Run on CUDA hardware only (TileLang lowers via NVRTC/NVCC).

References:
    https://github.com/tile-ai/tilelang/pull/746     (merged 2025-08-22)
    https://github.com/tile-ai/tilelang/blob/main/src/op/copy.cc
    cppmega/upstream_prs/08_tilelang_tma_bulk_copy_3d_smem_issue.md
"""
from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", v.split("+", 1)[0])
    return tuple(int(p) for p in parts[:3]) if parts else (0,)


def _build_kernel(tl, T, disable_tma: bool, shared_rank: int):
    """Build the kernel snippet from upstream_prs/08 template.

    shared_rank=3 reproduces the bug (3D alloc_shared + T.copy).
    shared_rank=2 is the pre-746 fast-path (baseline sanity).

    Uses TileLang's "lazy" style (inner @T.prim_func returned by outer
    function). Eager-style + out_idx was rejected in TileLang 0.1.8+ with
    `ValueError: out_idx is only supported in lazy mode`.
    """
    assert shared_rank in (2, 3)

    pass_configs = {
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: disable_tma,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }

    if shared_rank == 3:
        @tl.jit(out_idx=[-1], pass_configs=pass_configs)
        def build():
            @T.prim_func
            def k(
                X: T.Tensor([128, 4, 64], "bfloat16"),
                Y: T.Tensor([128, 4, 64], "bfloat16"),
            ):
                with T.Kernel(1) as (_,):
                    xs = T.alloc_shared([16, 4, 64], "bfloat16")  # 3D smem
                    T.copy(X[0:16, :, :], xs)
                    T.copy(xs, Y[0:16, :, :])
            return k
        return build

    @tl.jit(out_idx=[-1], pass_configs=pass_configs)
    def build():
        @T.prim_func
        def k(
            X: T.Tensor([128, 256], "bfloat16"),
            Y: T.Tensor([128, 256], "bfloat16"),
        ):
            with T.Kernel(1) as (_,):
                xs = T.alloc_shared([16, 256], "bfloat16")  # 2D smem
                T.copy(X[0:16, :], xs)
                T.copy(xs, Y[0:16, :])
        return k
    return build


_FATAL_PATTERNS = (
    re.compile(r"Cannot detect TMA layout", re.IGNORECASE),
    re.compile(r"Check failed:.*InputDim\(\)\s*==\s*2", re.IGNORECASE),
)


@contextmanager
def _capture_fd_stderr():
    """Redirect the process-level stderr (fd 2) to a temp file and yield its
    path. TileLang's `LOG(WARNING)` writes directly to the C stderr, which
    `redirect_stderr` (Python-level) does not capture."""
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield tmp
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


def _classify_exception(exc: BaseException, stderr_text: str) -> str:
    """Return 'regression' for the pre-746 hard-assert, 'other' for any other
    failure, 'ok' if we shouldn't have been called (no exception)."""
    blob = f"{type(exc).__name__}: {exc}\n{stderr_text}"
    for pat in _FATAL_PATTERNS:
        if pat.search(blob):
            return "regression"
    return "other"


def _compile_and_capture(tl, T, disable_tma: bool, shared_rank: int):
    """Compile the kernel and capture any stderr output (PR #746 uses
    LOG(WARNING) which goes to stderr). Returns (status, stderr, exc)."""
    import torch

    py_stderr = io.StringIO()
    py_stdout = io.StringIO()
    fd_stderr_bytes = b""
    try:
        with _capture_fd_stderr() as fd_err, \
                redirect_stderr(py_stderr), redirect_stdout(py_stdout):
            try:
                k = _build_kernel(tl, T, disable_tma=disable_tma, shared_rank=shared_rank)
                # Lazy JIT: call with the INPUT tensor only; output is allocated
                # by TileLang based on out_idx=[-1].
                dev = torch.device("cuda")
                if shared_rank == 3:
                    x = torch.randn(128, 4, 64, device=dev, dtype=torch.bfloat16)
                else:
                    x = torch.randn(128, 256, device=dev, dtype=torch.bfloat16)
                _ = k(x)
                status, exc = "ok", None
            except BaseException as e:  # noqa: BLE001 — catch ICHECK too
                fd_err.seek(0)
                fd_stderr_bytes = fd_err.read()
                combined_err = (
                    py_stderr.getvalue() + py_stdout.getvalue()
                    + fd_stderr_bytes.decode("utf-8", "replace")
                    + "\n" + traceback.format_exc()
                )
                return (_classify_exception(e, combined_err), combined_err, e)
            # Success path: still read fd stderr (for warn capture).
            fd_err.seek(0)
            fd_stderr_bytes = fd_err.read()
        combined = (
            py_stderr.getvalue() + py_stdout.getvalue()
            + fd_stderr_bytes.decode("utf-8", "replace")
        )
        return (status, combined, exc)
    except BaseException as exc:  # noqa: BLE001 — unexpected (fd redirection etc.)
        tb = traceback.format_exc()
        return ("other", tb, exc)


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: torch not importable", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required (TileLang lowers via NVRTC/NVCC).",
              file=sys.stderr)
        return 2

    try:
        import tilelang as tl
        import tilelang.language as T
    except ImportError as exc:
        print(f"ERROR: tilelang import failed: {exc}", file=sys.stderr)
        return 2

    tl_ver = getattr(tl, "__version__", "unknown")
    dev_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"TILELANG_VERSION : {tl_ver}")
    print(f"DEVICE           : {dev_name} (sm_{cc[0]}{cc[1]})")
    print(f"PR #746 (LowerBulkCopy warn+fallback) merged upstream 2025-08-22.")
    print(f"EXPECTED         : warn-not-assert on rank-3 smem (PR #746 behavior).")
    print(f"REGRESSION MARKER: any '{_FATAL_PATTERNS[0].pattern}' → exit 1.")
    print()

    # Suppress TileLang autotuner spam on stderr; we still capture it per-case.
    os.environ.setdefault("TVM_LOG_DEBUG", "0")

    regressed = False

    # --- Case A: rank-3 smem, TMA lowering ENABLED (the bug path). ---
    print("[A] rank-3 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=False")
    statusA, errA, excA = _compile_and_capture(tl, T, disable_tma=False, shared_rank=3)
    if statusA == "regression":
        print("    STATUS: REGRESSION — hard-assert on 'Cannot detect TMA layout'")
        print("    This is the pre-PR-746 failure. TileLang version is older than")
        print("    commit 5c11d24 (PR #746) or a regression has landed.")
        print("    Captured output (tail):")
        for line in errA.strip().splitlines()[-15:]:
            print(f"      {line}")
        regressed = True
    elif statusA == "other":
        print(f"    STATUS: UNEXPECTED FAIL — {type(excA).__name__}: {excA}")
        print("    Not the PR-746 assertion, but also not a clean compile. Tail:")
        for line in errA.strip().splitlines()[-15:]:
            print(f"      {line}")
        regressed = True
    else:
        low = errA.lower()
        fallback_hit = "fallback to normal copy" in low
        ws_skipped = "[ws] skipped" in low and "no tma copies" in low
        if fallback_hit:
            marker = "warn+fallback logged (PR #746 active)"
        elif ws_skipped:
            marker = "warp-spec skipped (confirms 3D copy took non-TMA path)"
        else:
            marker = "no warn captured (TMA may have succeeded for rank-3)"
        print(f"    STATUS: OK — compile succeeded ({marker})")
        for line in errA.strip().splitlines():
            low_line = line.lower()
            if ("tma" in low_line or "fallback" in low_line
                    or "[ws]" in low_line):
                print(f"      {line.strip()}")
    print()

    # --- Case B: rank-3 smem, TMA lowering DISABLED (baseline). ---
    print("[B] rank-3 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=True  (baseline)")
    statusB, errB, excB = _compile_and_capture(tl, T, disable_tma=True, shared_rank=3)
    if statusB != "ok":
        print(f"    STATUS: FAIL — {type(excB).__name__}: {excB}")
        print("    With TMA disabled the normal-copy path should always work.")
        for line in errB.strip().splitlines()[-10:]:
            print(f"      {line}")
        regressed = True
    else:
        print("    STATUS: OK")
    print()

    # --- Case C: rank-2 smem, TMA lowering ENABLED (fast-path sanity). ---
    print("[C] rank-2 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=False (fast-path)")
    statusC, errC, excC = _compile_and_capture(tl, T, disable_tma=False, shared_rank=2)
    if statusC != "ok":
        print(f"    STATUS: FAIL — {type(excC).__name__}: {excC}")
        print("    Rank-2 smem is the TMA fast-path, it must compile.")
        for line in errC.strip().splitlines()[-10:]:
            print(f"      {line}")
        regressed = True
    else:
        print("    STATUS: OK — rank-2 TMA fast-path compiled cleanly")
    print()

    print("=" * 72)
    if regressed:
        print("VERDICT: REGRESSION. TileLang's rank-3+ smem handling is broken")
        print("         (either the pre-746 hard-assert is back, or a new fault).")
        return 1
    print("VERDICT: OK. PR #746 warn+fallback behavior intact on",
          f"tilelang {tl_ver}.")
    print("         (Note: 3D smem still falls back to cp.async — TMA 3D")
    print("          descriptor support remains a separate feature request.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
