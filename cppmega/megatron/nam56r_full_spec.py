"""Full NAM56R stack spec substrate for cppmega Megatron lanes."""

from __future__ import annotations

import copy
import os
from typing import Iterable

from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.experimental_attention_variant_module_specs import get_dsa_module_spec_for_backend
from megatron.core.models.mamba.mamba_layer_specs import mamba_inference_stack_spec, mamba_stack_spec
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_layer import TransformerLayer

from cppmega.megatron.author_mamba3_spec import AuthorMamba3Mixer
from cppmega.megatron.dsa_local_spec import get_cppmega_dsa_layer_spec
from cppmega.megatron.m2rnn_spec import CppMegaM2RNNMixer
from cppmega.megatron.nam56r_layout import (
    FULL_NAM56R_DEPTH,
    FULL_NAM56R_PATTERN,
    load_attention_layer_numbers,
    load_dsa_a_layer_ranks,
    load_r_layer_indices,
)
from cppmega.megatron.mla_shared import (
    build_attention_layer_spec_from_self_attention_spec,
    build_mla_attention_layer_spec,
)
from cppmega.recipes.nam56r_launch import build_nam56r_lite_main_pattern

# NO FALLBACKS: TE spec provider is required for NAM56R full spec.
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

try:
    from megatron.core.extensions.kitchen import KitchenSpecProvider
except ImportError:
    KitchenSpecProvider = None  # Kitchen is optional (experimental Megatron extension)



def _get_backend_spec_provider(config):
    assert config.transformer_impl == "transformer_engine", (
        "CPPMega mixed MLA/DSA only supports transformer_engine implementation for now."
    )
    if getattr(config, "use_kitchen", False):
        if KitchenSpecProvider is None or TESpecProvider is None:
            raise ImportError("Kitchen backend requested but Kitchen/TE provider is unavailable")
        return KitchenSpecProvider(
            fallback=TESpecProvider(),
            use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
            kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
        )
    if TESpecProvider is None:
        raise ImportError("Transformer Engine backend provider is unavailable")
    return TESpecProvider()


def _clone_attention_variant_config(config, *, experimental_attention_variant):
    """Keep mixed MLA/DSA branch behavior local to the selected attention layer."""

    cloned = copy.copy(config)
    try:
        cloned.experimental_attention_variant = experimental_attention_variant
    except AttributeError:
        return config
    return cloned


class CppMegaSelectiveMambaMixer(nn.Module):
    """Select Author Mamba3 or M2RNN per global Mamba-like layer position."""

    def __init__(
        self,
        config,
        d_model: int,
        submodules=None,
        layer_number: int | None = None,
        pg_collection=None,
        pp_layer_offset: int = 0,
        r_layer_indices: Iterable[int] = (),
    ) -> None:
        super().__init__()
        indices = frozenset(int(index) for index in r_layer_indices)
        layer_idx = 1 if layer_number is None else layer_number
        mixer_cls = CppMegaM2RNNMixer if layer_idx in indices else AuthorMamba3Mixer
        self.impl = mixer_cls(
            config=config,
            d_model=d_model,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )

    def forward(self, *args, **kwargs):
        return self.impl(*args, **kwargs)

    def mamba_state_shapes_per_request(self):
        return self.impl.mamba_state_shapes_per_request()


class CppMegaSelectiveAttentionLayer(TransformerLayer):
    """Select MLA or upstream DSA on A-layers by A-layer rank."""

    def __init__(
        self,
        config,
        submodules=None,
        layer_number: int | None = None,
        pg_collection=None,
        add_layer_offset: bool = False,
        pp_layer_offset: int = 0,
        is_mtp_layer: bool = False,
        dsa_a_layer_ranks: Iterable[int] = (),
        attention_layer_numbers: Iterable[int] = (),
    ) -> None:
        del submodules
        dsa_a_layer_ranks = frozenset(int(x) for x in dsa_a_layer_ranks)
        attention_layer_numbers = tuple(int(x) for x in attention_layer_numbers)
        layer_idx = 1 if layer_number is None else int(layer_number)
        try:
            a_rank = attention_layer_numbers.index(layer_idx)
        except ValueError as exc:
            raise ValueError(f"layer_number={layer_idx} is not an A-layer in the NAM56R pattern") from exc
        if a_rank in dsa_a_layer_ranks:
            branch_config = _clone_attention_variant_config(
                config, experimental_attention_variant="dsa"
            )
            self_attention_spec = get_dsa_module_spec_for_backend(
                config=branch_config, backend=_get_backend_spec_provider(branch_config)
            )
            spec = build_attention_layer_spec_from_self_attention_spec(branch_config, self_attention_spec)
        else:
            branch_config = _clone_attention_variant_config(
                config, experimental_attention_variant=None
            )
            spec = build_mla_attention_layer_spec(branch_config)
        super().__init__(
            config=branch_config,
            submodules=spec.submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
        )



def build_cppmega_nam56r_full_stack_spec(config):
    r_layer_indices = load_r_layer_indices()
    attention_layer_numbers = load_attention_layer_numbers()
    upstream = (
        mamba_inference_stack_spec.submodules
        if config.transformer_impl == "inference_optimized"
        else mamba_stack_spec.submodules
    )
    attention_layer = ModuleSpec(
        module=CppMegaSelectiveAttentionLayer,
        params={
            "dsa_a_layer_ranks": load_dsa_a_layer_ranks(),
            "attention_layer_numbers": attention_layer_numbers,
        },
    )
    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    norm=WrappedTorchNorm,
                    mixer=ModuleSpec(
                        module=CppMegaSelectiveMambaMixer,
                        params={"r_layer_indices": r_layer_indices},
                    ),
                    mamba_bda=get_bias_dropout_add,
                ),
            ),
            gdn_layer=upstream.gdn_layer,
            attention_layer=attention_layer,
            mlp_layer=upstream.mlp_layer,
            moe_layer=upstream.moe_layer,
            mtp_block_spec=upstream.mtp_block_spec,
        ),
    )



def build_default_hybrid_layer_pattern(*, mtp_depths: int = 1) -> str:
    pattern = os.environ.get("CPPMEGA_NEM_PATTERN", FULL_NAM56R_PATTERN)
    depth = int(os.environ.get("CPPMEGA_LAYER_DEPTH", str(FULL_NAM56R_DEPTH)))
    return build_nam56r_lite_main_pattern(pattern=pattern, depth=depth, mtp_depths=mtp_depths)
