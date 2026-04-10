"""NAM56R-lite Mamba stack spec for the real H200 training lane."""

from cppmega.megatron.mla_shared import build_mla_attention_layer_spec
from cppmega.megatron.nam56r_full_spec import (
    CppMegaSelectiveMambaMixer,
    build_cppmega_nam56r_full_stack_spec,
    build_default_hybrid_layer_pattern,
    load_r_layer_indices,
)



def build_cppmega_nam56r_lite_stack_spec(config):
    return build_cppmega_nam56r_full_stack_spec(config)
