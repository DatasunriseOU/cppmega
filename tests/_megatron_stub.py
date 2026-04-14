"""Shared helper: install a ``megatron`` stub with a *real* ``ModuleSpec``.

Why this exists
---------------
Several of our tests probe for megatron at collection time via
``importlib.util.find_spec("megatron")`` so they can ``@pytest.mark.skipif``
the heavyweight tests that need the real package.

On Python 3.12+, ``importlib.util.find_spec`` asserts that the module's
``__spec__`` attribute is a real ``importlib.machinery.ModuleSpec``.  A
plain ``MagicMock()`` installed into ``sys.modules['megatron']`` (as several
tests in this suite used to do) exposes a ``MagicMock`` for ``__spec__``
which fails that type check with::

    ValueError: megatron.__spec__ is not set

This helper installs a MagicMock into ``sys.modules`` but attaches a real
``ModuleSpec(name, loader=None)`` so ``find_spec`` returns a usable spec
object instead of raising.  It is idempotent: if ``megatron`` already looks
like a real install (``sys.modules['megatron'].__spec__`` is a real
``ModuleSpec``) it leaves things alone.

Usage from a test module::

    from tests._megatron_stub import install_megatron_stub
    install_megatron_stub()   # before any ``importlib.util.find_spec("megatron")``
"""
from __future__ import annotations

import importlib.machinery
import sys
from unittest.mock import MagicMock


_DEFAULT_SUBMODULES = (
    "megatron",
    "megatron.core",
    "megatron.core.tensor_parallel",
    "megatron.core.transformer",
    "megatron.core.transformer.module",
    "megatron.core.transformer.transformer_config",
    "megatron.core.transformer.spec_utils",
    "megatron.core.models",
    "megatron.core.models.mamba",
)


def install_megatron_stub(submodules: tuple[str, ...] = _DEFAULT_SUBMODULES) -> None:
    """Install a MagicMock stub for megatron + submodules with real ModuleSpecs.

    If ``megatron`` is already a real import (its ``__spec__`` is a real
    ``ModuleSpec``), this is a no-op.  Otherwise every listed submodule gets
    a MagicMock in ``sys.modules`` with ``__spec__`` set to a real
    ``ModuleSpec(name, loader=None)`` so that ``importlib.util.find_spec``
    succeeds on Python 3.12+.
    """
    existing = sys.modules.get("megatron")
    if existing is not None and isinstance(
        getattr(existing, "__spec__", None), importlib.machinery.ModuleSpec
    ):
        # Already a real (or previously-stubbed-correctly) megatron; leave it.
        return

    root = MagicMock()
    for name in submodules:
        mod = root
        for part in name.split(".")[1:]:
            mod = getattr(mod, part)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        # Mark the module so ``is_real_megatron_available`` can distinguish
        # our stub from a genuine megatron install.
        mod.__cppmega_stub__ = True
        sys.modules[name] = mod

    # Some tests also need MegatronModule to be a real nn.Module subclass
    # so downstream class-level code (``class Foo(MegatronModule):``) works.
    try:
        import torch  # noqa: WPS433  (local import: torch optional at collection)
        root.core.transformer.module.MegatronModule = torch.nn.Module
    except ImportError:
        pass


def is_real_megatron_available() -> bool:
    """Return True iff the installed ``megatron`` is the real package.

    ``importlib.util.find_spec("megatron")`` is insufficient because our
    MagicMock stub (installed by ``install_megatron_stub``) carries a real
    ``ModuleSpec`` for Python 3.12 compatibility — so ``find_spec`` returns
    a valid spec even for the stub.  Check for our sentinel flag instead.
    """
    try:
        spec = importlib.util.find_spec("megatron")
    except ValueError:
        return False
    if spec is None:
        return False
    mod = sys.modules.get("megatron")
    if mod is None:
        return True  # find_spec found something, not imported yet
    return not getattr(mod, "__cppmega_stub__", False)


# Re-export for convenience.
import importlib.util  # noqa: E402  (needed by is_real_megatron_available)
