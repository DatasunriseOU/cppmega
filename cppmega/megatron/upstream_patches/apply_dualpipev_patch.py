"""Apply DualPipeV integration to Megatron training.

Wires ``cppmega.megatron.dualpipev_schedule.setup_dualpipev_from_args``
into Megatron's ``setup_model_and_optimizer`` so the model is wrapped
in a DualPipeV 2-rank pipeline after construction and before the first
training step.

Activated by env var ``CPPMEGA_DUALPIPEV=1``.  Requires:

- Megatron started with ``--pipeline-model-parallel-size 1``
- World size a multiple of 2 (DualPipeV pairs ranks)
- ``dualpipe`` package installed (``pip install git+https://github.com/deepseek-ai/DualPipe.git``)
- NAM56R 52-layer model layout

Failure mode: crashes loudly on any mismatch — no silent fallback.
"""
from __future__ import annotations

import os
from typing import Any, Callable


_ORIGINAL_SETUP: Callable | None = None
_DUALPIPEV_STATE: Any | None = None


def _patched_setup_model_and_optimizer(*args, **kwargs):
    """Call upstream ``setup_model_and_optimizer``, then wrap the model
    in DualPipeV before returning to the training loop.

    The upstream signature is ``(model_provider_func, model_type, ...)``
    returning ``(model, optimizer, opt_param_scheduler)``.  We pass
    through unchanged, then attach DualPipeV to the returned ``model``.
    """
    global _DUALPIPEV_STATE
    assert _ORIGINAL_SETUP is not None
    model, optimizer, scheduler = _ORIGINAL_SETUP(*args, **kwargs)

    # Parse megatron args — the first positional is usually model_provider_func
    # so we can't rely on it.  Use the global args instead.
    from megatron.training import get_args

    margs = get_args()

    from cppmega.megatron.dualpipev_schedule import setup_dualpipev_from_args

    # ``model`` is usually a list of DDP-wrapped modules in Megatron.
    if isinstance(model, list):
        if len(model) != 1:
            raise RuntimeError(
                f"DualPipeV: expected a single model chunk at PP=1, got "
                f"{len(model)} chunks — is VPP enabled?"
            )
        model_ref = model[0]
    else:
        model_ref = model

    _DUALPIPEV_STATE = setup_dualpipev_from_args(model_ref, margs)
    print(
        "[cppmega] DualPipeV activated: pipe_rank={}, dp_size={}, "
        "num_chunks={}, pipe_groups_carved".format(
            _DUALPIPEV_STATE.groups.pipe_rank,
            _DUALPIPEV_STATE.groups.dp_size,
            _DUALPIPEV_STATE.num_chunks,
        ),
        flush=True,
    )

    return model, optimizer, scheduler


def apply() -> None:
    """Install the DualPipeV hook.  Idempotent; safe to call multiple times.

    Gate: only installs if ``CPPMEGA_DUALPIPEV=1``.  Otherwise no-op, so
    production (default-off) is unaffected.
    """
    if os.environ.get("CPPMEGA_DUALPIPEV", "0") != "1":
        return

    global _ORIGINAL_SETUP
    if _ORIGINAL_SETUP is not None:
        return  # already installed

    try:
        import megatron.training.training as _t
    except ImportError as e:
        raise RuntimeError(
            "DualPipeV patch: cannot import megatron.training.training"
        ) from e

    # Also verify dualpipe is available — fail fast.
    try:
        import dualpipe  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "DualPipeV patch: dualpipe package not installed.  Install with "
            "`pip install git+https://github.com/deepseek-ai/DualPipe.git`"
        ) from e

    _ORIGINAL_SETUP = _t.setup_model_and_optimizer
    _t.setup_model_and_optimizer = _patched_setup_model_and_optimizer
    print("[cppmega] DualPipeV hook installed on setup_model_and_optimizer", flush=True)


def revert() -> None:
    """Undo the hook.  Used in tests / when toggling the gate."""
    global _ORIGINAL_SETUP, _DUALPIPEV_STATE
    if _ORIGINAL_SETUP is None:
        return
    import megatron.training.training as _t

    _t.setup_model_and_optimizer = _ORIGINAL_SETUP
    _ORIGINAL_SETUP = None
    _DUALPIPEV_STATE = None


def get_state():
    """Return the current DualPipeVState (or None if not activated)."""
    return _DUALPIPEV_STATE


if __name__ == "__main__":
    # Manual trigger for debugging.
    apply()
