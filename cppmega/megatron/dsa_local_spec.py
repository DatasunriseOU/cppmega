"""Helpers for running official Megatron DSA in the cppmega smoke lane."""

from __future__ import annotations

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
)


def get_cppmega_dsa_layer_spec(config, backend=None):
    """Return the upstream Megatron DSA module spec unchanged.

    `cppmega` should validate the official DSA surface first before copying any
    nanochat sparse-attention code.
    """

    return get_dsa_module_spec_for_backend(config=config, backend=backend)
