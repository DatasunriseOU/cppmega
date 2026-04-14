"""Reproducer: TileLang `LayoutInference` raises `tvm.error.InternalError:
Divide by zero` (FloorMod const-fold) when a kernel uses `csr % R` / `csr // R`
indexing inside a `T.Parallel` loop together with TMA lowering enabled.

Fires inside this call chain (reproduced end-to-end below):

    tvm::tl::LayoutInferencer::Substitute
      → BufferUseDefCollector::Run
      → ParallelOpNode::InferLayout
      → ParallelOpNode::CompleteBufferFragment
      → tvm::tl::Fragment::Fragment
      → tvm::tl::infer_fragment_index
      → tvm::tl::MakeFlattenedExpression
      → tvm::arith::NormalizeIterMapToExpr
      → IterMapToExprNormalizer::ConvertIterSplitExpr
      → tvm::floormod(PrimExpr, PrimExpr, Span)
      → tvm::arith::TryConstFold<tvm::tir::FloorMod>
      → tvm.error.InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero

Context: The Mamba3 MIMO `bwd_bwd` kernel in state-spaces/mamba originally
held rank-3 shared-memory operands that tripped TileLang's `LowerBulkCopy`
`InputDim() == 2` assertion (separate issue, covered by TileLang PR #746 with
a WARNING+fallback). A layout-fix patch flattens those 3D buffers to 2D and
introduces `q_frag[csr, n] += q_bias_frag[csr % R, n]` inside `T.Parallel`.
With TMA lowering enabled (`TL_DISABLE_TMA_LOWER=False`) the flattened
kernel exercises a code path in LayoutInference that hits FloorMod const-fold
on an intermediate where the constant value is `0`, crashing the compiler.

The minimal isolated pattern does NOT reproduce (kernel is too small for
LayoutInference to construct the problematic iter-map). The bug only fires
at the full `mamba_mimo_bwd_bwd_kernel` structure. This reproducer therefore
reconstructs that exact structure by:

  1. Locating the installed `mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd`
     source file (state-spaces/mamba upstream),
  2. Applying the included `mamba3_bwd_layout_fix.patch` (the 2D-smem flatten),
  3. Flipping the PassConfig `TL_DISABLE_TMA_LOWER` / `TL_DISABLE_WARP_SPECIALIZED`
     in the already-decorated `@tilelang.jit` from `True` to `False`,
  4. Loading the resulting module and invoking `mamba_mimo_bwd_bwd` at a tiny
     NAM56R-compatible shape (R=4, chunk_size=16, B=1, S=64, H=4, G=1, N=64, P=64),
  5. Catching the `tvm.error.InternalError` and printing
     `TILELANG_BUG_REPRODUCED` with the full backtrace.

Exit code:
  0 — bug is NOT present (kernel compiles or unrelated crash)
  1 — bug IS present (FloorMod const-fold divide-by-zero in LayoutInference)
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT_FIX_PATCH = os.path.join(HERE, "mamba3_bwd_layout_fix.patch")


def _locate_upstream_bwd() -> str:
    """Return absolute path of the installed mamba_ssm mamba3_mimo_bwd.py."""
    try:
        import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd as mod
    except Exception as e:
        raise SystemExit(
            f"ERROR: could not import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd: {e}\n"
            "Install state-spaces/mamba (e.g. `pip install mamba-ssm` or from source).\n"
        )
    path = getattr(mod, "__file__", None)
    if not path:
        raise SystemExit("ERROR: mamba3_mimo_bwd has no __file__.")
    return path


def _prepare_patched_module(src_path: str) -> str:
    """Apply layout-fix patch + flip TMA/WS flags into a /tmp copy; return path."""
    work = tempfile.mkdtemp(prefix="tilelang_floormod_repro_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src_path, dst)

    # Apply the layout-fix patch (flattens 3D→2D smem, introduces `csr%R` pattern).
    # We patch in the work dir so `-p4` correctly resolves the path strip.
    # Patch paths inside: `a/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`
    # → 4 leading components to strip to match our local "mamba3_mimo_bwd.py".
    res = subprocess.run(
        ["patch", "-p4", dst],
        input=open(LAYOUT_FIX_PATCH, "rb").read(),
        capture_output=True,
        cwd=work,
    )
    if res.returncode != 0:
        print("patch stdout:", res.stdout.decode(errors="replace"))
        print("patch stderr:", res.stderr.decode(errors="replace"))
        raise SystemExit(
            "ERROR: `patch -p4` failed to apply mamba3_bwd_layout_fix.patch.\n"
            "This usually means the installed mamba_ssm source doesn't match "
            "the expected upstream layout. Pin to state-spaces/mamba main "
            "around 2026-04 (the patch was generated against that snapshot)."
        )

    # Flip TMA/WS flags in the @tilelang.jit decorator (both bwd_fwd + bwd_bwd).
    src = open(dst).read()
    src2 = (
        src.replace(
            "tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,",
            "tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,",
        ).replace(
            "tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,",
            "tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,",
        )
    )
    if src2 == src:
        raise SystemExit("ERROR: could not find TMA/WS PassConfig flags to flip.")
    open(dst, "w").write(src2)
    return dst


def _import_from_path(mod_name: str, path: str):
    # Ensure parent namespace package is importable so relative imports resolve.
    import mamba_ssm.ops.tilelang.mamba3  # noqa: F401
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    # Skip at import time if CUDA isn't configured — tilelang still runs
    # the LayoutInference pass before codegen, so the crash is observable
    # on any machine with `tilelang` + `tvm-ffi` installed, even without
    # a working NVCC (LayoutInference runs entirely on host).
    try:
        import tilelang  # noqa: F401
    except Exception as e:
        print(f"ERROR: tilelang not importable: {e}")
        return 2

    if not os.path.exists(LAYOUT_FIX_PATCH):
        print(f"ERROR: missing patch file {LAYOUT_FIX_PATCH}")
        return 2

    src_path = _locate_upstream_bwd()
    print(f"Upstream mamba3_mimo_bwd.py: {src_path}")

    patched = _prepare_patched_module(src_path)
    print(f"Patched + TMA-flipped copy:   {patched}")

    mod = _import_from_path("mamba3_bwd_floormod_repro", patched)
    mamba_mimo_bwd_bwd = mod.mamba_mimo_bwd_bwd

    # Tiny NAM56R-compatible shape. Enough to exercise the LayoutInference
    # path that hits TryConstFold<FloorMod>.
    B, S, H, G, N, P, R = 1, 64, 4, 1, 64, 64, 4
    print(
        f"Compiling mamba_mimo_bwd_bwd at "
        f"B={B} S={S} H={H} G={G} N={N} P={P} R={R} (TMA+WS = ON)..."
    )

    try:
        kernel = mamba_mimo_bwd_bwd(
            B, S, H, G, N, P, R,
            hasZ=False, hasD=False, reduceO=False,
            chunk_size=16, rotary_dim_divisor=4, dtype="float16",
            threads=256, num_stages=0,
        )
        # `@tilelang.jit` returns a JITFunc; getattr get_kernel_source forces compile.
        if hasattr(kernel, "get_kernel_source"):
            out = kernel.get_kernel_source()
            print(f"OK: compiled cleanly (CUDA source {len(out)} chars).")
        else:
            print(f"OK: compiled — returned {type(kernel).__name__}.")
        print()
        print("TILELANG_BUG_NOT_REPRODUCED")
        return 0

    except Exception as e:  # noqa: BLE001
        msg = str(e)
        # Capture the traceback frames — the FloorMod / LayoutInference
        # origin is encoded in the C++ frames, not in the InternalError text.
        tb_str = traceback.format_exc()
        print()
        print(f"CRASH: {type(e).__name__}")
        print(textwrap.shorten(msg, width=600))
        print()
        print(tb_str)
        combined = (msg + "\n" + tb_str).lower()
        if "divide by zero" in combined and (
            "floormod" in combined
            or "tryconstfold" in combined
            or "layoutinference" in combined
        ):
            print()
            print("TILELANG_BUG_REPRODUCED: LayoutInference FloorMod divide-by-zero")
            return 1
        if "inputdim() == 2" in combined or "cannot detect tma layout" in combined:
            print()
            print(
                "DIFFERENT BUG: LowerBulkCopy InputDim==2 assert — this is the "
                "sibling issue already tracked in upstream_prs/08. Not this bug."
            )
            return 2
        print()
        print("UNEXPECTED_CRASH: neither FloorMod DBZ nor known InputDim==2")
        return 2


if __name__ == "__main__":
    sys.exit(main())
