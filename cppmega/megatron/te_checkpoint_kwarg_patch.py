"""Compatibility patch for Megatron TE checkpoint call kwargs.

Recent Megatron TransformerLayer code calls ``te_checkpoint`` as:

    te_checkpoint(apply_module(self.mlp), ..., hidden_states, padding_mask=mask)

Some Transformer Engine / PyTorch combinations forward that call kwarg down to
``torch.utils.checkpoint.checkpoint()``, which rejects arbitrary kwargs before
step 1.  The older working NAM path passed the same value through
``functools.partial(self.mlp, padding_mask=mask)``.  This patch restores that
calling convention at Megatron's ``te_checkpoint`` boundary while preserving
real checkpoint-control kwargs.
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType
from typing import Any

from cppmega.megatron.document_isolation import bind_current_structure_batch


_PATCH_FLAG = "_cppmega_te_checkpoint_kwarg_patch_applied"
_ORIGINAL_ATTR = "_cppmega_te_checkpoint_original"

# kwargs consumed by TE/PyTorch checkpoint itself rather than by the model
# function.  Everything else is treated as a call kwarg and captured in the
# checkpointed forward callable.
_CHECKPOINT_CONTROL_KWARGS = frozenset(
    {
        "context_fn",
        "determinism_check",
        "debug",
        "preserve_rng_state",
        "use_reentrant",
    }
)


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    control_kwargs: dict[str, Any] = {}
    call_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _CHECKPOINT_CONTROL_KWARGS:
            control_kwargs[key] = value
        else:
            call_kwargs[key] = value
    return control_kwargs, call_kwargs


def apply_te_checkpoint_kwarg_patch(module: ModuleType | Any | None = None) -> bool:
    """Patch ``megatron.core.extensions.transformer_engine.te_checkpoint``.

    Returns ``True`` when the patch is installed or was already installed.
    Raises when the target function is unavailable; FP8+MLP recompute should not
    silently proceed on the known-broken path.
    """

    if module is None:
        module = importlib.import_module("megatron.core.extensions.transformer_engine")

    if getattr(module, _PATCH_FLAG, False):
        return True

    original = getattr(module, "te_checkpoint")

    @functools.wraps(original)
    def _cppmega_te_checkpoint(
        forward_func,
        distribute_saved_activations,
        get_rng_state_tracker,
        tp_group,
        *args,
        **kwargs,
    ):
        if kwargs:
            control_kwargs, call_kwargs = _split_kwargs(dict(kwargs))
            if call_kwargs:
                forward_func = functools.partial(forward_func, **call_kwargs)
            kwargs = control_kwargs
        forward_func = bind_current_structure_batch(forward_func)
        return original(
            forward_func,
            distribute_saved_activations,
            get_rng_state_tracker,
            tp_group,
            *args,
            **kwargs,
        )

    setattr(_cppmega_te_checkpoint, _ORIGINAL_ATTR, original)
    setattr(module, "te_checkpoint", _cppmega_te_checkpoint)
    setattr(module, _PATCH_FLAG, True)
    print("[cppmega-te-checkpoint] Patched Megatron te_checkpoint kwarg forwarding", flush=True)
    return True
