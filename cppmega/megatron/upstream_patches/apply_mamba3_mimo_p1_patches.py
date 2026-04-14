"""Apply Mamba3 MIMO P1 optimization patches: TMA lower + warp specialization.

Plan P1 from `reference_mamba_ssm_optimization_plan.md`:
  Flip the two `TL_DISABLE_*` pass_config flags in the upstream Mamba3 MIMO
  TileLang kernels from True -> False, so the TileLang compiler emits
  TMA + warp-specialized code. Also ensures that
  `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` is present on every kernel so the
  sm_121 (GB10) bwd_bwd smem 140KB-vs-100KB blocker stays resolved.

This patch is **opt-in** via the env var `CPPMEGA_MAMBA3_P1=1`. Import of this
module does NOT patch anything. Call `apply_all()` explicitly or run the
module as a script.

The patch is idempotent — safe to run multiple times. After applying it
verifies each target file contains the expected flipped flags and raises
loudly if the expected line was not found (no silent fallback).

Scope (8 sites across 4 files):
  * mamba3_mimo_fwd.py         ->  mamba_mimo_fwd (line ~34, 35)
  * mamba3_mimo_bwd.py         ->  mamba_mimo_bwd_fwd (line ~38, 39)
                                  mamba_mimo_bwd_bwd (line ~501, 502)
  * mamba3_mimo_fwd_varlen.py  ->  mamba_mimo_fwd (line ~55, 56)
  * mamba3_mimo_bwd_varlen.py  ->  mamba_mimo_bwd_fwd (line ~58, 59)
                                  mamba_mimo_bwd_bwd (line ~540, 541)

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

# Line that follows the disable block — we use it to safely inject
# TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE when missing.
_FAST_MATH_LINE = "        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,\n"
_AGGRESSIVE_MERGE_LINE = (
    "        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,  # cppmega P1 GB10 smem\n"
)

# Expected per-file occurrence count of the disable block.
_EXPECTED_SITES: dict[str, int] = {
    "mamba3_mimo_fwd.py": 1,
    "mamba3_mimo_bwd.py": 2,
    "mamba3_mimo_fwd_varlen.py": 1,
    "mamba3_mimo_bwd_varlen.py": 2,
}


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


def _patch_one_file(path: Path) -> tuple[int, int]:
    """Patch a single kernel file.

    Returns (sites_patched, sites_with_aggressive_merge_added).

    Raises RuntimeError if the expected disable block is not found the
    expected number of times AND the file isn't already patched.
    """
    if not path.exists():
        raise RuntimeError(f"Kernel file missing: {path}")

    text = path.read_text()
    name = path.name
    expected = _EXPECTED_SITES[name]

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
    path.write_text(new_text)
    print(f"  DONE {name}: {expected} site(s) flipped to TMA+WarpSpec=True")

    aggressive_added = _ensure_aggressive_merge(path, already_loaded=new_text)
    return expected, aggressive_added


def _ensure_aggressive_merge(path: Path, already_loaded: str | None = None) -> int:
    """Make sure every `pass_configs={...}` dict in the file contains
    `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`.

    This is required so GB10 (sm_121) bwd_bwd still fits in 100KB smem.
    """
    text = already_loaded if already_loaded is not None else path.read_text()

    expected = _EXPECTED_SITES[path.name]
    already = text.count("TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE")
    if already >= expected:
        return 0

    # Inject the aggressive-merge line right after every FAST_MATH line that
    # doesn't already have it immediately following. We walk occurrences so
    # we only add when missing.
    new_lines: list[str] = []
    added = 0
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        new_lines.append(lines[i])
        if lines[i] == _FAST_MATH_LINE:
            # Look ahead: is the aggressive-merge line already the next line?
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE" not in nxt:
                new_lines.append(_AGGRESSIVE_MERGE_LINE)
                added += 1
        i += 1

    if added == 0:
        return 0

    new_text = "".join(new_lines)
    path.write_text(new_text)
    print(
        f"  DONE {path.name}: added TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE "
        f"to {added} site(s) (GB10 smem guard)"
    )
    return added


def apply_all() -> None:
    """Apply P1 patches to the live mamba_ssm installation."""
    base = _find_mamba3_tilelang_dir()
    print(f"Mamba3 TileLang path: {base}")
    print()

    total_flipped = 0
    total_aggr_added = 0
    for name in _EXPECTED_SITES:
        flipped, aggr = _patch_one_file(base / name)
        total_flipped += flipped
        total_aggr_added += aggr

    print()
    print(
        f"P1 patches complete: {total_flipped} disable-block flip(s), "
        f"{total_aggr_added} aggressive-merge line(s) inserted."
    )
    print("Restart training to recompile kernels with TMA + warp specialization.")


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
