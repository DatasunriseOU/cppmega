"""Remove stale cutlass-dsl / tvm-ffi / flash-attn namespace packages.

The base GHCR image ships nvidia_cutlass_dsl as a *namespace package* (a bare
directory in dist-packages without proper __init__.py ownership). When
flash-attn-4==4.0.0b23 is pip-installed on top, Python still resolves the OLD
namespace directory first, shadowing the wheel-bundled cutlass-dsl and breaking
FA4 imports. pip uninstall cannot remove namespace packages, and shell globs /
rm -rf are unreliable inside Modal's image builder due to quoting issues.

This standalone script uses only pathlib + shutil (no shell globs) so it can be
copied into the image and run as a single, simple command:

    python3 /opt/fix_cutlass_namespace.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Prefixes (lowercased) of packages/dirs that must be purged before the fresh
# FA4 beta23 install. Matched against both underscore and dash spellings.
PATTERNS = [
    "nvidia_cutlass_dsl",
    "cutlass",
    "tvm_ffi",
    "apache_tvm_ffi",
    "flash_attn",
]


def find_dist_packages() -> list[Path]:
    """Locate candidate dist-packages / site-packages directories."""
    candidates: list[Path] = []
    for base in (
        Path("/usr/local/lib/python3.13/dist-packages"),
        Path("/usr/local/lib/python3.12/dist-packages"),
        Path("/usr/local/lib/python3.11/dist-packages"),
    ):
        if base.is_dir():
            candidates.append(base)
    # Fall back to whatever the running interpreter actually uses.
    for raw in sys.path:
        p = Path(raw)
        if p.is_dir() and (p.name in {"dist-packages", "site-packages"}):
            if p not in candidates:
                candidates.append(p)
    return candidates


def matches(name: str) -> bool:
    low = name.lower()
    for pat in PATTERNS:
        if low.startswith(pat) or low.startswith(pat.replace("_", "-")):
            return True
    return False


def clean_base(base: Path) -> list[str]:
    removed: list[str] = []
    for item in base.iterdir():
        if not matches(item.name):
            continue
        try:
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed.append(item.name)
        except OSError as exc:  # pragma: no cover - best effort cleanup
            print(f"WARN: could not remove {item}: {exc}", file=sys.stderr)
    return removed


def verify(base: Path) -> list[str]:
    """Return any leftover entries that should have been removed."""
    leftovers: list[str] = []
    if not base.is_dir():
        return leftovers
    for item in base.iterdir():
        if matches(item.name):
            leftovers.append(item.name)
    return leftovers


def main() -> int:
    bases = find_dist_packages()
    if not bases:
        print("No dist-packages/site-packages directory found; nothing to do.")
        return 0

    total_removed: list[str] = []
    for base in bases:
        removed = clean_base(base)
        total_removed.extend(removed)
        print(f"[{base}] removed {len(removed)} items: {removed[:10]}")

    print(f"Removed {len(total_removed)} items total.")

    # Verify nothing matching the patterns survived.
    problems: list[str] = []
    for base in bases:
        leftovers = verify(base)
        if leftovers:
            problems.extend(f"{base}/{name}" for name in leftovers)

    if problems:
        print(f"VERIFY FAILED: leftovers remain: {problems}", file=sys.stderr)
        return 1

    print("VERIFY OK: no cutlass-dsl/tvm-ffi/flash-attn leftovers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
