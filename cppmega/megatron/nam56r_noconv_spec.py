"""NAM56R stack spec on the vanilla Mamba-2 SSD path (no conv1d, with Mamba3 B/C features).

This is the Branch-B alternative to ``nam56r_te_spec``: instead of wrapping the
Author Mamba3 module (which calls ``mamba3_siso_combined`` / ``mamba3_mimo_combined``
and was measured at 127k tok/sec / ~30% MFU on H200x8 for the mamba3_te recipe),
this spec uses ``NoConvMamba3BCMixer`` -- the Megatron-native ``MambaMixer`` flow
built on ``mamba_chunk_scan_combined`` (the same kernel as vanilla Mamba-2 SSD at
211k tok/sec / 50% MFU) with:

  - conv1d removed (Mamba3 convention, RoPE not yet re-added)
  - QK-Norm on B and C  (Mamba3 feature)
  - Learnable B/C bias  (Mamba3 feature)
  - No trapezoidal / data-dep A / RoPE / MIMO  (deferred; bugs found in the
    full ``Mamba3NoConvMixer`` variant -- see docs/changelog.md)

Rationale
~~~~~~~~~

The mamba3_te recipe using Author kernels hits 127k/~30% MFU; the gap to the
211k/50% baseline is dominated by (a) the Author kernels not participating in
TE CUDA graph fusion and (b) preprocessing overhead on top of a slower scan.
By routing the SSM through the SAME kernel as the vanilla baseline and
applying QK-Norm + bias in lightweight Python preprocessing, we should land
close to the 166k tok/sec already measured for ``CppMegaMamba3Mixer``
(which keeps conv1d) while stripping the conv1d stage entirely.

The feature surface intentionally matches the user's 127k recipe description
("да QK-Norm, B/C bias") so performance comparisons are apples-to-apples.
"""

from __future__ import annotations

from typing import Iterable

from torch import nn

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from cppmega.megatron.m2rnn_spec import CppMegaM2RNNMixer
from cppmega.megatron.nam56r_layout import has_megatron_dsa_symbol, load_r_layer_indices
from cppmega.megatron.noconv_mamba_mixer import (
    NoConvMamba3BCMixer,
    NoConvMambaMixerSubmodules,
)


class CppMegaNoConvSelectiveMambaMixer(nn.Module):
    """Select ``NoConvMamba3BCMixer`` (M-layer) or ``CppMegaM2RNNMixer`` (R-layer).

    Mirrors ``CppMegaSelectiveMambaMixerTE`` from ``nam56r_te_spec`` but routes
    M-layers through the vanilla SSD kernel + Mamba3 B/C features instead of
    through the Author Mamba3 kernels.  R-layer selection logic is unchanged.
    """

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
        indices = frozenset(int(i) for i in r_layer_indices)
        layer_idx = 1 if layer_number is None else int(layer_number)
        if layer_idx in indices:
            self.impl = CppMegaM2RNNMixer(
                config=config,
                d_model=d_model,
                submodules=submodules,
                layer_number=layer_number,
                pg_collection=pg_collection,
                pp_layer_offset=pp_layer_offset,
            )
        else:
            # ``NoConvMamba3BCMixer`` takes ``(config, submodules, d_model, ...)``
            # positionally; pass via kwargs so the Mamba3 B/C mixer can duck-type
            # on ``MambaMixerSubmodules`` (same ``in_proj`` / ``out_proj`` shape
            # as ``NoConvMambaMixerSubmodules``).
            self.impl = NoConvMamba3BCMixer(
                config=config,
                submodules=submodules,
                d_model=d_model,
                layer_number=layer_number,
                pg_collection=pg_collection,
                pp_layer_offset=pp_layer_offset,
            )

    def forward(self, *args, **kwargs):
        return self.impl(*args, **kwargs)

    def mamba_state_shapes_per_request(self):
        return self.impl.mamba_state_shapes_per_request()


def build_cppmega_nam56r_noconv_stack_spec(config):
    """Build NAM56R spec using upstream TE submodules + NoConvMamba3BCMixer.

    Same TE-fused submodules as ``build_cppmega_nam56r_te_stack_spec`` (norm,
    attention, MLP, MoE, MTP all upstream-TE).  Only the Mamba mixer changes
    to route through the vanilla SSD scan kernel with Mamba3 B/C preprocessing.

    R-layer positions (12/24/36/48 in the 52-layer NAM56R layout) are always
    routed through ``CppMegaM2RNNMixer``, which dispatches to the fused Triton
    ``m2rnn_scan_triton`` kernel (see ``cppmega/megatron/m2rnn_triton.py``).
    The learned M2RNN state transition is part of the NAM56R architecture and
    cannot be silently disabled.
    """
    r_layer_indices = load_r_layer_indices()
    upstream = mamba_stack_spec.submodules
    upstream_mamba_sub = upstream.mamba_layer.submodules

    submodules_kwargs = dict(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                # Keep the upstream Mamba-layer norm (TE fused into in_proj)
                norm=upstream_mamba_sub.norm,
                # Replace ONLY the mixer with the noconv selector
                mixer=ModuleSpec(
                    module=CppMegaNoConvSelectiveMambaMixer,
                    # ``NoConvMamba3BCMixer`` reads ``submodules.in_proj`` /
                    # ``submodules.out_proj`` -- identical field names to
                    # Megatron's ``MambaMixerSubmodules``, so we reuse the
                    # upstream TE-fused linear specs.
                    submodules=upstream_mamba_sub.mixer.submodules,
                    params={"r_layer_indices": r_layer_indices},
                ),
                mamba_bda=upstream_mamba_sub.mamba_bda,
            ),
        ),
        # All other layers unchanged from upstream (TE-fused)
        gdn_layer=upstream.gdn_layer,
        attention_layer=upstream.attention_layer,
        mlp_layer=upstream.mlp_layer,
        moe_layer=upstream.moe_layer,
        mtp_block_spec=upstream.mtp_block_spec,
    )

    # PR #3553: provide dsa_layer when available.
    if has_megatron_dsa_symbol():
        submodules_kwargs["dsa_layer"] = getattr(
            upstream, "dsa_layer", upstream.attention_layer
        )

    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(**submodules_kwargs),
    )


# Alias for ``--spec cppmega.megatron.nam56r_noconv_spec cppmega_nam56r_noconv_stack_spec``
cppmega_nam56r_noconv_stack_spec = build_cppmega_nam56r_noconv_stack_spec


__all__ = [
    "CppMegaNoConvSelectiveMambaMixer",
    "build_cppmega_nam56r_noconv_stack_spec",
    "cppmega_nam56r_noconv_stack_spec",
    "NoConvMambaMixerSubmodules",
]
