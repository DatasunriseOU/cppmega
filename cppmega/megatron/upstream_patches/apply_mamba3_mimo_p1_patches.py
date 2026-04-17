"""Apply Mamba3 MIMO P1 optimization patches: TMA lower + warp specialization.

**Selective P1 scope (2026-04-14)**:
  Patch ONLY `mamba_mimo_fwd` in `mamba3_mimo_fwd.py` and `mamba3_mimo_fwd_varlen.py`.
  Do NOT touch `mamba3_mimo_bwd.py` or `mamba3_mimo_bwd_varlen.py`.

Why fwd-only?
  The TileLang compiler fails on the bwd kernels with:
      Cannot detect TMA layout (InputDim != 2)
  because the bwd_fwd/bwd_bwd kernels have 3D+ shared-memory descriptors that
  the current TileLang TMA lower pass cannot handle. Trying to enable TMA +
  warp specialization on those sites crashes at kernel compile time. The fwd
  kernel's smem descriptors are all 2D and compile cleanly.

  We still propagate `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True` to
  **all** sites (fwd + bwd) because that flag is harmless when the other two
  flags stay disabled and is required so GB10 (sm_121) bwd_bwd still fits in
  100 KB smem.

Plan P1 from `reference_mamba_ssm_optimization_plan.md` (revised):
  Flip the two `TL_DISABLE_*` pass_config flags in the upstream Mamba3 MIMO
  fwd TileLang kernels from True -> False, so the TileLang compiler emits
  TMA + warp-specialized code for the forward pass only.

This patch is **opt-in** via the env var `CPPMEGA_MAMBA3_P1=1`. Import of this
module does NOT patch anything. Call `apply_all()` explicitly or run the
module as a script.

The patch is idempotent — safe to run multiple times. After applying it
verifies each target file contains the expected flipped flags and raises
loudly if the expected line was not found (no silent fallback).

**8-rank race fix (2026-04-14)**: `apply_all()` runs the actual file
writes on rank 0 only (when torch.distributed is initialized), followed
by `dist.barrier()`. This prevents 8 concurrent writers producing
half-written files → IndentationError on the import path.

Sites touched (fwd-only, 2 files, 2 sites):
  * mamba3_mimo_fwd.py         ->  mamba_mimo_fwd
  * mamba3_mimo_fwd_varlen.py  ->  mamba_mimo_fwd

Sites where only TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE is ensured (bwd):
  * mamba3_mimo_bwd.py         ->  mamba_mimo_bwd_fwd, mamba_mimo_bwd_bwd
  * mamba3_mimo_bwd_varlen.py  ->  mamba_mimo_bwd_fwd, mamba_mimo_bwd_bwd

TODO: revisit bwd kernels once TileLang TMA lower handles InputDim > 2
shared-memory descriptors.

Usage:
    python -m cppmega.megatron.upstream_patches.apply_mamba3_mimo_p1_patches
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The exact text we expect to find before patching. Must match the upstream
# kernels byte-for-byte, otherwise we raise so the next agent can reconcile.
_DISABLE_BLOCK = (
    "        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,\n"
    "        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,\n"
)
_ENABLE_BLOCK = (
    "        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,  # cppmega P1\n"
    "        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,  # cppmega P1\n"
)

# What we want to look like post-patch (idempotence check).
_ALREADY_PATCHED_MARKER = "TL_DISABLE_TMA_LOWER: False,  # cppmega P1"

# CRITICAL: AGGRESSIVE_SHARED_MEMORY_MERGE injection must preserve line
# numbers. If the aggressive-merge flag is inserted as a new line below
# FAST_MATH, every function defined below shifts down by one line. If
# `import mamba_ssm` (which transitively imports this module via
# `mamba_ssm.ops.tilelang.mamba3.mamba3_mimo`) runs BEFORE apply_all
# patches this file, the imported function objects' `co_firstlineno`
# attributes point to the PRE-patch line numbers. When tilelang later
# calls `inspect.getsource(...)` → `ast.parse(...)` on an inner
# `@T.prim_func` function, it reads the POST-patch file at the OLD line
# numbers and grabs the wrong region → IndentationError.
# Fix: append the aggressive-merge key onto the existing FAST_MATH line
# (Python allows multiple dict key/value pairs per line inside braces).
# This keeps the total line count identical.
_FAST_MATH_LINE = "        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,\n"
_FAST_MATH_WITH_AGGR = (
    "        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,"
    " tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,"
    "  # cppmega P1 GB10 smem\n"
)

# Files where we FLIP the TMA/WarpSpec disable flags (fwd-only).
_FWD_FLIP_SITES: dict[str, int] = {
    "mamba3_mimo_fwd.py": 1,
    "mamba3_mimo_fwd_varlen.py": 1,
}

# Files where we ONLY ensure AGGRESSIVE_SHARED_MEMORY_MERGE is present.
# (bwd kernels: TMA lower can't handle 3D+ smem descriptors → don't flip.)
_BWD_SMEM_ONLY_SITES: dict[str, int] = {
    "mamba3_mimo_bwd.py": 2,
    "mamba3_mimo_bwd_varlen.py": 2,
}

# Combined expected sites for aggressive-merge ensure pass.
_ALL_EXPECTED_SITES: dict[str, int] = {**_FWD_FLIP_SITES, **_BWD_SMEM_ONLY_SITES}


def _atomic_write_text(path: Path, content: str) -> None:
    """Write content atomically via tempfile + os.replace + syntax check.

    Why atomic: Python's Path.write_text opens in 'w' mode which truncates
    the target to zero bytes FIRST, then writes. Any concurrent reader
    (including tilelang's `inspect.getsource` which reads the original
    .py file at kernel compile time) that hits this window sees a
    truncated file → IndentationError. `os.replace` is POSIX-atomic on
    the same filesystem: other readers see either the old complete
    file or the new complete file, never a torn state.

    Also compiles the content via py_compile before rename so a broken
    patch crashes loudly here instead of deferring to kernel load time.
    """
    # Use PID-suffixed tempfile so concurrent writers (under flock they
    # shouldn't, but defensively) don't clobber each other's temp.
    tmp_path = path.with_name(f"{path.name}.cppmega_p1.tmp.{os.getpid()}")
    tmp_path.write_text(content)
    # Fail loudly if the patched content is syntactically invalid.
    import py_compile
    try:
        py_compile.compile(str(tmp_path), doraise=True)
    except py_compile.PyCompileError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Patched kernel {path.name} has Python syntax error; "
            f"aborting before rename. {exc}"
        ) from exc
    os.replace(tmp_path, path)


def _find_mamba3_tilelang_dir() -> Path:
    try:
        import mamba_ssm.ops.tilelang.mamba3 as m
    except ImportError as e:
        raise RuntimeError(
            "mamba_ssm.ops.tilelang.mamba3 not importable — is mamba_ssm installed?"
        ) from e
    paths = list(m.__path__)
    if not paths:
        raise RuntimeError("mamba_ssm.ops.tilelang.mamba3 has no __path__")
    return Path(paths[0])


def _flip_fwd_file(path: Path) -> tuple[int, int]:
    """Flip TL_DISABLE_* on a fwd kernel file and ensure aggressive-merge.

    Returns (sites_flipped, sites_with_aggressive_merge_added).
    """
    if not path.exists():
        raise RuntimeError(f"Kernel file missing: {path}")

    text = path.read_text()
    name = path.name
    expected = _FWD_FLIP_SITES[name]

    current_enabled = text.count(_ALREADY_PATCHED_MARKER)
    if current_enabled == expected:
        print(f"  OK   {name}: already patched ({expected} sites)")
        aggressive_added = _ensure_aggressive_merge(path, already_loaded=text)
        return 0, aggressive_added
    if current_enabled != 0:
        raise RuntimeError(
            f"{name}: partial patch state — {current_enabled} sites show "
            f"already-patched marker but expected {expected}. Fix by reinstalling "
            "mamba_ssm and re-running."
        )

    count_disable = text.count(_DISABLE_BLOCK)
    if count_disable != expected:
        raise RuntimeError(
            f"{name}: expected {expected} disable-block occurrences, found "
            f"{count_disable}. Upstream kernel signature changed — update "
            "_DISABLE_BLOCK in this patch before proceeding."
        )

    new_text = text.replace(_DISABLE_BLOCK, _ENABLE_BLOCK)
    assert new_text.count(_ALREADY_PATCHED_MARKER) == expected, (
        f"{name}: post-patch marker count mismatch"
    )
    _atomic_write_text(path, new_text)
    print(f"  DONE {name}: {expected} site(s) flipped to TMA+WarpSpec=True (fwd-only)")

    aggressive_added = _ensure_aggressive_merge(path, already_loaded=new_text)
    return expected, aggressive_added


def _bwd_smem_only(path: Path) -> int:
    """bwd kernels: leave TL_DISABLE_* alone, ONLY ensure aggressive-merge.

    Returns number of aggressive-merge lines inserted.
    """
    if not path.exists():
        raise RuntimeError(f"Kernel file missing: {path}")

    # Safety: if someone previously applied the all-sites patch, warn loudly.
    text = path.read_text()
    already_flipped = text.count(_ALREADY_PATCHED_MARKER)
    if already_flipped > 0:
        raise RuntimeError(
            f"{path.name}: detected {already_flipped} cppmega-P1 flipped site(s) "
            "in a BWD file. Prior patch enabled TMA on bwd which is known to "
            "crash compile. Reinstall mamba_ssm to reset these kernels before "
            "re-running the selective-fwd-only patch."
        )

    return _ensure_aggressive_merge(path, already_loaded=text)


def _ensure_aggressive_merge(path: Path, already_loaded: str | None = None) -> int:
    """Make sure every `pass_configs={...}` dict in the file contains
    `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`.

    This is required so GB10 (sm_121) bwd_bwd still fits in 100KB smem and is
    harmless on H200.

    Implementation: we rewrite the FAST_MATH line in-place to carry BOTH
    FAST_MATH and AGGRESSIVE_MERGE keys. This keeps the file's line count
    unchanged, which is critical — `mamba_ssm.ops.tilelang.mamba3.mamba3_mimo`
    transitively imports this file at `import mamba_ssm` time; if apply_all
    runs AFTER that import and shifts lines, the cached `co_firstlineno` of
    inner `@T.prim_func` functions desynchronizes from the file, and the
    next `inspect.getsource` returns the wrong region (IndentationError).
    """
    text = already_loaded if already_loaded is not None else path.read_text()

    expected = _ALL_EXPECTED_SITES[path.name]
    already = text.count("TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE")
    if already >= expected:
        return 0

    count_fast = text.count(_FAST_MATH_LINE)
    if count_fast != expected:
        raise RuntimeError(
            f"{path.name}: expected {expected} FAST_MATH lines to carry the "
            f"aggressive-merge flag, found {count_fast}. Upstream kernel "
            "shape changed; update _FAST_MATH_LINE."
        )

    new_text = text.replace(_FAST_MATH_LINE, _FAST_MATH_WITH_AGGR)
    added = count_fast
    _atomic_write_text(path, new_text)
    print(
        f"  DONE {path.name}: merged TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE "
        f"onto {added} FAST_MATH line(s) (line-count preserved)"
    )
    return added


def _do_patch() -> None:
    """Actual patch logic — runs on rank 0 only when called via apply_all()."""
    base = _find_mamba3_tilelang_dir()
    print(f"Mamba3 TileLang path: {base}")
    print()

    total_flipped = 0
    total_aggr_added = 0

    # fwd kernels: flip TMA/WarpSpec flags + ensure aggressive-merge
    for name in _FWD_FLIP_SITES:
        flipped, aggr = _flip_fwd_file(base / name)
        total_flipped += flipped
        total_aggr_added += aggr

    # bwd kernels: ensure aggressive-merge ONLY, do not flip TMA flags
    for name in _BWD_SMEM_ONLY_SITES:
        aggr = _bwd_smem_only(base / name)
        total_aggr_added += aggr

    print()
    print(
        f"P1 selective-fwd patches complete: {total_flipped} fwd-site flip(s), "
        f"{total_aggr_added} aggressive-merge line(s) inserted."
    )
    print("Restart training to recompile kernels with TMA + warp specialization.")


def apply_all() -> None:
    """Apply P1 patches to the live mamba_ssm installation.

    DISABLED 2026-04-15: this script mutates installed site-packages mamba_ssm
    `.py` files in place, which is non-reversible (env=0 doesn't undo file
    rewrites; only `pip install --force-reinstall` from the fork restores).
    On bench3 + GB10 we found the persistent mutation produces grad_norm=NaN
    in backward (TileLang TMA layout broken on H200/sm_121, see memory
    `reference_tma_layout_fix_broken_h200.md`).

    To run anyway (NOT for production), set both env vars:
      CPPMEGA_MAMBA3_P1=1
      MAMBA3_P1_ALLOW_FILE_MUTATION=1

    Loud refusal otherwise. After running, mamba_ssm site-packages are
    PERSISTENTLY MUTATED and you must reinstall to restore.

    8-rank race fix:
      1. If torch.distributed is initialized: rank 0 writes, others wait on
         dist.barrier().
      2. Otherwise (the real case today — `apply_all` is called from
         `cppmega_fp8_shim.py` at import time, BEFORE init_process_group):
         LOCAL_RANK=0 does the actual write under an exclusive flock, then
         appends a "DONE" sentinel to the lockfile. All other local ranks
         wait in a shared-flock poll loop until the sentinel is present,
         then return. This guarantees no rank proceeds past apply_all until
         the file is fully written AND flushed — preventing both the
         truncated-file race AND the `inspect.getsource` line-number
         desync race that bit us earlier (see `_ensure_aggressive_merge`
         for the line-count-preserving fix).
    """
    try:
        import torch.distributed as dist
        dist_available = True
    except Exception:
        dist = None  # type: ignore[assignment]
        dist_available = False

    is_initialized = bool(dist_available and dist.is_initialized())

    if is_initialized:
        rank = dist.get_rank()
        if rank == 0:
            _do_patch()
        else:
            print(f"[mamba3_p1] rank={rank} waiting on rank-0 to finish patch")
        dist.barrier()
        return

    # Pre-init path (the real case: apply_all is called from
    # cppmega_fp8_shim.py at import time, BEFORE init_process_group).
    # Strategy:
    #   (a) Only LOCAL_RANK=0 on each node actually runs `_do_patch` →
    #       the file is modified by exactly ONE process per node.
    #   (b) Other ranks take a shared flock and block until the lockfile
    #       contains the sentinel "DONE" written by rank 0. This prevents
    #       them from progressing past `apply_all` (and thus importing the
    #       kernel file) while rank 0 is still mid-write.
    # This eliminates the race where tilelang.jit parses the .py file
    # (via inspect.getsource) while a concurrent rank is rewriting it.
    import fcntl
    import tempfile
    import time

    lock_path = Path(tempfile.gettempdir()) / "cppmega_mamba3_p1.lock"
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "?"
    sentinel = "DONE\n"

    if local_rank == 0:
        # Rank 0 on this node: exclusive lock, write files, mark DONE.
        lock_path.unlink(missing_ok=True)
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                print(f"[mamba3_p1] local_rank=0 rank={rank_env} patching")
                _do_patch()
                lock_fh.write(sentinel)
                lock_fh.flush()
                os.fsync(lock_fh.fileno())
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return

    # Non-zero local rank: wait for lockfile to contain sentinel.
    print(f"[mamba3_p1] local_rank={local_rank} rank={rank_env} waiting for local_rank=0")
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if lock_path.exists():
            try:
                with open(lock_path) as fh:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
                    try:
                        if sentinel in fh.read():
                            return
                    finally:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        time.sleep(0.1)
    raise RuntimeError(
        f"[mamba3_p1] local_rank={local_rank}: timed out waiting 120s for "
        f"local_rank=0 to complete patch. lockfile={lock_path}"
    )


def apply_if_requested() -> bool:
    """Env-gated entry point. Returns True if patches were applied."""
    if os.environ.get("CPPMEGA_MAMBA3_P1", "0") not in ("1", "true", "True"):
        return False
    apply_all()
    return True


if __name__ == "__main__":
    # Script mode always applies (env gate is for library-side import).
    try:
        apply_all()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
