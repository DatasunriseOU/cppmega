from __future__ import annotations

from types import SimpleNamespace

from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch


def test_te_checkpoint_patch_moves_call_kwargs_into_forward_partial():
    calls = []

    def original(forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args, **kwargs):
        calls.append(
            {
                "forward_func": forward_func,
                "distribute_saved_activations": distribute_saved_activations,
                "get_rng_state_tracker": get_rng_state_tracker,
                "tp_group": tp_group,
                "args": args,
                "kwargs": kwargs,
            }
        )
        return forward_func(*args)

    module = SimpleNamespace(te_checkpoint=original)
    apply_te_checkpoint_kwarg_patch(module)

    out = module.te_checkpoint(
        lambda hidden_states, *, padding_mask=None: (hidden_states, padding_mask),
        False,
        "rng",
        "tp",
        "hidden",
        padding_mask="mask",
    )

    assert out == ("hidden", "mask")
    assert calls[0]["args"] == ("hidden",)
    assert calls[0]["kwargs"] == {}


def test_te_checkpoint_patch_preserves_checkpoint_control_kwargs():
    calls = []

    def original(forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args, **kwargs):
        calls.append(kwargs)
        return forward_func(*args)

    module = SimpleNamespace(te_checkpoint=original)
    apply_te_checkpoint_kwarg_patch(module)

    out = module.te_checkpoint(
        lambda hidden_states, *, padding_mask=None: hidden_states,
        False,
        "rng",
        "tp",
        "hidden",
        padding_mask=None,
        determinism_check="none",
        debug=True,
    )

    assert out == "hidden"
    assert calls == [{"determinism_check": "none", "debug": True}]


def test_te_checkpoint_patch_is_idempotent():
    def original(forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args, **kwargs):
        return forward_func(*args, **kwargs)

    module = SimpleNamespace(te_checkpoint=original)

    assert apply_te_checkpoint_kwarg_patch(module) is True
    patched = module.te_checkpoint
    assert apply_te_checkpoint_kwarg_patch(module) is True
    assert module.te_checkpoint is patched
