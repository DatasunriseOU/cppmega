"""Megatron-specific cppmega glue modules."""

import os as _os


def _run_smem_preflight() -> None:
    """Optionally run the GB10 sm_121 smem preflight on import.

    Opt-in via ``CPPMEGA_SMEM_CHECK=1``.  This runs the static AST check
    against our tracked TileLang kernel files and raises
    ``SmemPreflightError`` if any kernel is missing
    ``TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE`` (hard fail on sm_121).

    Training launch scripts prefer ``python -m cppmega.megatron.preflight_smem_check``
    as an explicit pre-step; the env-gated auto-invocation here is a
    defense-in-depth for ad-hoc imports.
    """
    if _os.environ.get("CPPMEGA_SMEM_CHECK", "0") != "1":
        return
    # Imported lazily so that importing ``cppmega.megatron`` on machines
    # without torch / tilelang doesn't explode unless the check is
    # explicitly requested.
    from . import preflight_smem_check as _pf

    _pf.check(raise_on_error=True)


_run_smem_preflight()
