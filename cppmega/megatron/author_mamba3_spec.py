"""Author-pure Mamba3 seam wrapped in Megatron-local Mamba builders.

When used inside ``nam56r_te_spec`` the upstream ``MambaLayer.norm`` is
``IdentityOp`` (a no-op) because the standard Megatron mixer fuses
LayerNorm into ``TELayerNormColumnParallelLinear``.  Since the Author
``Mamba3`` module uses a plain ``nn.Linear`` for its ``in_proj`` (no
fused norm), we must apply RMSNorm explicitly in this mixer so the
residual stream is normalised before projection.

Without this norm the residual magnitudes grow unbounded through 52
layers, producing NaN grad_norm from iteration 1.
"""

from __future__ import annotations

import torch
from torch import nn

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from cppmega.features.mamba3 import build_author_mamba3_config
from cppmega.megatron.mamba_local_spec import build_cppmega_local_stack_spec


def _group_world_size(group) -> int:
    if group is None:
        return 1
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


class AuthorMamba3Mixer(nn.Module):
    """Training-only Megatron mixer shim around the upstream author Mamba3.

    Includes an explicit RMSNorm before ``Mamba3.in_proj`` to compensate
    for the missing fused norm when used with ``IdentityOp`` as the
    ``MambaLayer`` norm (standard in ``mamba_stack_spec``).
    """

    def __init__(
        self,
        config: TransformerConfig,
        d_model: int,
        submodules=None,
        layer_number: int | None = None,
        pg_collection=None,
        pp_layer_offset: int = 0,
    ) -> None:
        super().__init__()
        del submodules, pp_layer_offset

        if pg_collection is None:
            raise ValueError("pg_collection must be provided for AuthorMamba3Mixer")

        tp_world_size = _group_world_size(pg_collection.tp)
        cp_world_size = _group_world_size(getattr(pg_collection, "cp", None))
        if tp_world_size != 1:
            raise NotImplementedError(
                "AuthorMamba3Mixer currently supports tensor-model-parallel-size=1 only"
            )
        if cp_world_size != 1:
            raise NotImplementedError(
                "AuthorMamba3Mixer currently supports context-parallel-size=1 only"
            )
        try:
            from mamba_ssm.modules.mamba3 import Mamba3
        except ImportError as exc:
            raise ImportError(
                "Author Mamba3 is not installed. Re-run scripts/remote_setup_h200.sh with "
                "INSTALL_AUTHOR_MAMBA3=1."
            ) from exc

        author_config = build_author_mamba3_config(config, d_model=d_model)
        device = None
        if torch.cuda.is_available() and not getattr(config, "use_cpu_initialization", False):
            device = torch.cuda.current_device()

        # RMSNorm applied before in_proj to compensate for IdentityOp norm
        # in MambaLayer.  Matches the behaviour of TELayerNormColumnParallelLinear
        # which fuses LayerNorm + Linear in the standard MambaMixer path.
        self.pre_norm = nn.RMSNorm(
            d_model,
            eps=getattr(config, "layernorm_epsilon", 1e-5),
            device=device,
            dtype=config.params_dtype,
        )

        self.mixer = Mamba3(
            d_model=author_config.d_model,
            d_state=author_config.d_state,
            expand=author_config.expand,
            headdim=author_config.headdim,
            ngroups=author_config.ngroups,
            rope_fraction=author_config.rope_fraction,
            dt_min=author_config.dt_min,
            dt_max=author_config.dt_max,
            dt_init_floor=author_config.dt_init_floor,
            A_floor=author_config.A_floor,
            is_outproj_norm=author_config.is_outproj_norm,
            is_mimo=author_config.is_mimo,
            mimo_rank=author_config.mimo_rank,
            chunk_size=author_config.chunk_size,
            layer_idx=None if layer_number is None else max(layer_number - 1, 0),
            n_layer=getattr(config, "num_layers", None),
            device=device,
            dtype=config.params_dtype,
        )

    def forward(
        self,
        hidden_states,
        inference_context=None,
        *,
        inference_params=None,
        packed_seq_params=None,
    ):
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("AuthorMamba3Mixer does not support Megatron inference paths yet")
        if packed_seq_params is not None:
            raise NotImplementedError("AuthorMamba3Mixer does not support packed sequences yet")
        if hidden_states.ndim != 3:
            raise ValueError("AuthorMamba3Mixer expects hidden_states shaped [seq, batch, hidden]")

        # Apply RMSNorm before projection (compensates for IdentityOp norm
        # in upstream MambaLayer).
        hidden_states = self.pre_norm(hidden_states)

        batch_first = hidden_states.transpose(0, 1).contiguous()
        out = self.mixer(batch_first)
        if out.shape != batch_first.shape:
            raise RuntimeError(
                f"AuthorMamba3Mixer returned shape {tuple(out.shape)} for input {tuple(batch_first.shape)}"
            )
        return out.transpose(0, 1).contiguous(), None

    def mamba_state_shapes_per_request(self):
        raise NotImplementedError("AuthorMamba3Mixer does not support Megatron inference cache shapes yet")


cppmega_author_mamba3_stack_spec = build_cppmega_local_stack_spec(
    mamba_mixer_spec=ModuleSpec(module=AuthorMamba3Mixer)
)
