"""Root pytest conftest for cppmega.

Fixes a collection error seen on bench3/europe H200 machines when pytest runs
the full ``tests/`` suite in one invocation:

    ImportError while loading conftest ...
    AttributeError: module 'megatron' has no attribute '__spec__'
    (or: megatron.__spec__ is not set)

Root cause: ``megatron`` is a *namespace package* split across
``megatron/core/`` (Megatron-Core) and the local ``cppmega/megatron/`` shim
package that re-exports helpers under the same top-level name. Namespace
packages created implicitly via ``pkgutil.extend_path`` / lazy imports can
end up with ``__spec__ = None`` if the first importer of ``megatron`` is
pytest's assertion-rewrite loader before the real package finishes setup.
Python 3.12+ started enforcing ``module.__spec__`` on reload, which trips
pytest rewrite.

The fix is to eagerly import ``megatron`` once at the session root, let the
real package register itself with ``sys.modules`` with a valid ``ModuleSpec``,
and leave it alone. We swallow ImportError so this file is also safe on
machines without Megatron installed (e.g. local dev Macs) — those tests will
fail individually with a clear ``No module named 'torch'`` / ``megatron``
error, not a cascading collection abort.
"""

from __future__ import annotations

import importlib
import sys


def _ensure_module_spec(name: str) -> None:
    """Import ``name`` eagerly and make sure ``__spec__`` is populated.

    Some namespace-package configurations leave ``module.__spec__`` as
    ``None`` which breaks pytest's assertion rewriting on Python 3.12+.
    If that happens, fall back to ``importlib.util.find_spec`` to attach a
    real spec manually.
    """
    try:
        mod = importlib.import_module(name)
    except ImportError:
        # Module genuinely not installed — individual tests that need it
        # will fail with a clear error. Don't mask that at collection time.
        return

    if getattr(mod, "__spec__", None) is None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            mod.__spec__ = spec
            sys.modules[name] = mod


# Order matters: import megatron.core first so the real package wins over
# any implicit namespace package created by cppmega.megatron.* re-exports.
for _name in ("megatron", "megatron.core"):
    _ensure_module_spec(_name)
