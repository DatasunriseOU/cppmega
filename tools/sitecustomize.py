"""Optional local Python startup hook for cppmega tools.

This file is intentionally inert unless CPPMEGA_MEMORY_DTYPE_AUDIT=1 is set.
It lets the memory dtype audit hook install through PYTHONPATH without editing
the GB10 launch scripts.
"""

from __future__ import annotations

import os
import sys


if os.environ.get("CPPMEGA_MEMORY_DTYPE_AUDIT", "0") == "1":
    import builtins

    _orig_import = builtins.__import__
    _installed = False

    def _try_install() -> None:
        global _installed
        if _installed or "megatron.training.training" not in sys.modules:
            return
        _installed = True
        try:
            import memory_dtype_audit

            memory_dtype_audit.install()
            builtins.__import__ = _orig_import
        except Exception as exc:  # pragma: no cover - startup hook must not crash training
            print(
                f"[dtype_audit] sitecustomize failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):
        module = _orig_import(name, globals, locals, fromlist, level)
        if name.startswith("megatron.training"):
            _try_install()
        return module

    builtins.__import__ = _import_hook
