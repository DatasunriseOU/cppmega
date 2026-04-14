"""Reproducer: Megatron-LM fused linear CE dispatcher rejects every non-Blackwell CUDA arch.

File under test (on Megatron-LM `dev` branch):
    megatron/core/fusions/fused_linear_cross_entropy.py
    class Platform.__init__  (~line 30-43)

It raises ``ValueError(f"Unsupported architecture: {cc[0]}")`` for every
compute capability whose major != 10 (i.e. everything except Blackwell
SM100). That includes H100 / H200 (cc=9.0), A100 (8.0), L40/Ada (8.9),
GB10 (12.1), etc.

This script:
  1. Prints environment (torch version, megatron-core version, GPU cc).
  2. Imports ``_get_platform`` and calls it.
  3. On Hopper (cc[0]==9) or any non-Blackwell CUDA device: expects the
     ValueError to fire — exit code 0 means BUG REPRODUCED.
  4. On Blackwell (cc[0]==10): expects no exception — native path works,
     exit code 0 means EXPECTED.
  5. On non-CUDA hosts (darwin/CPU-only) or pre-#3345 megatron trees that
     lack the file: exit code 77 (SKIP) with a diagnostic message.

After PR #3345 merges, Hopper will stop raising; update this script's
"expected" arm to treat cc=9 the same as cc=10.

Exit codes:
    0  — bug correctly reproduced (ValueError on non-Blackwell) OR
         native path works (on Blackwell or post-#3345 Hopper).
    1  — unexpected behavior (e.g. Blackwell raised, or non-Blackwell
         silently succeeded without the fix).
    77 — SKIP: cannot run (no CUDA, or megatron-core too old).

References:
    - Open PR:   https://github.com/NVIDIA/Megatron-LM/pull/3345
    - Dispatcher file (dev branch):
      https://github.com/NVIDIA/Megatron-LM/blob/dev/megatron/core/fusions/fused_linear_cross_entropy.py
    - Our reroute:
      /Volumes/external/sources/cppmega/cppmega/megatron/apply_linear_ce_patch.py
"""
from __future__ import annotations

import platform
import sys
import traceback


def _print_env() -> None:
    print(f"[env] platform={platform.platform()}")
    print(f"[env] python={sys.version.split()[0]}")
    try:
        import torch

        print(f"[env] torch={torch.__version__}")
        print(f"[env] cuda.is_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            cc = torch.cuda.get_device_capability(dev)
            name = torch.cuda.get_device_name(dev)
            print(f"[env] cuda.cc={cc}  device={name!r}  device_idx={dev}")
    except Exception as exc:  # pragma: no cover
        print(f"[env] torch import failed: {exc}")

    try:
        import megatron.core as mc

        print(f"[env] megatron-core={getattr(mc, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"[env] megatron-core import failed: {exc}")


def main() -> int:
    _print_env()

    try:
        import torch
    except ImportError:
        print("[skip] torch not installed; cannot exercise dispatcher.")
        return 77

    if not torch.cuda.is_available():
        print("[skip] no CUDA device; dispatcher asserts CUDA availability first.")
        return 77

    cc = torch.cuda.get_device_capability(torch.cuda.current_device())

    # Pull the exact symbol used by LinearCrossEntropyModule internally.
    # Presence of this module implies we are on Megatron-LM dev (post-PR #2206)
    # or a release that ships it (>=0.17 expected).
    try:
        from megatron.core.fusions.fused_linear_cross_entropy import _get_platform
    except ImportError as exc:
        print(
            f"[skip] megatron.core.fusions.fused_linear_cross_entropy not "
            f"importable ({exc}). This means the megatron-core install pre-dates "
            f"PR #2206 (no dispatcher). Bug cannot be reproduced here."
        )
        return 77

    print(f"[reproducer] calling _get_platform() on cc={cc} …")

    try:
        _get_platform()
    except ValueError as exc:
        msg = str(exc)
        print(f"[reproducer] caught ValueError: {msg!r}")
        if "Unsupported architecture" not in msg:
            print(f"[fail] wrong ValueError body; expected 'Unsupported architecture'.")
            return 1
        if cc[0] == 10:
            print(
                "[fail] Blackwell (cc=10) hit the 'Unsupported architecture' "
                "path — dispatcher is broken."
            )
            return 1
        print(
            f"[ok] BUG REPRODUCED on cc={cc}. "
            f"Dispatcher rejects every non-Blackwell CUDA device. "
            f"Fix: land PR #3345 (adds cc=9 native Hopper path) and/or add a "
            f"soft fallback instead of raising for other cc values."
        )
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[fail] unexpected exception type: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    # No exception: native kernel is installed.
    if cc[0] == 10:
        print(f"[ok] cc={cc} (Blackwell) — native Blackwell entry loaded (expected).")
        return 0
    if cc[0] == 9:
        print(
            f"[ok] cc={cc} (Hopper) — native Hopper entry loaded. "
            f"This means PR #3345 has been applied to this tree."
        )
        return 0

    print(
        f"[fail] cc={cc} reached a non-raising branch that isn't Blackwell or "
        f"Hopper. Dispatcher has an unexpected path; investigate."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
