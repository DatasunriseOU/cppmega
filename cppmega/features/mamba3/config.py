"""Small, testable config mapping for the author Mamba3 seam."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthorMamba3Config:
    d_model: int
    d_state: int
    expand: int
    headdim: int
    ngroups: int
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    is_outproj_norm: bool = False
    is_mimo: bool = False
    mimo_rank: int = 4
    chunk_size: int = 64


def build_author_mamba3_config(megatron_config, *, d_model: int | None = None) -> AuthorMamba3Config:
    """Map the Megatron Mamba config surface onto the author Mamba3 contract."""

    hidden_size = getattr(megatron_config, "hidden_size", None) if d_model is None else d_model
    state_dim = getattr(megatron_config, "mamba_state_dim", None)
    head_dim = getattr(megatron_config, "mamba_head_dim", None)
    num_groups = getattr(megatron_config, "mamba_num_groups", None)
    expand = getattr(megatron_config, "mamba_expand", 2)

    if hidden_size is None or hidden_size <= 0:
        raise ValueError("author Mamba3 wrapper requires a positive hidden_size")
    if state_dim is None or state_dim <= 0:
        raise ValueError("author Mamba3 wrapper requires a positive mamba_state_dim")
    if head_dim is None or head_dim <= 0:
        raise ValueError("author Mamba3 wrapper requires a positive mamba_head_dim")
    if num_groups is None or num_groups <= 0:
        raise ValueError("author Mamba3 wrapper requires a positive mamba_num_groups")
    if expand <= 0:
        raise ValueError("author Mamba3 wrapper requires a positive expand factor")

    d_inner = hidden_size * expand
    if d_inner % head_dim != 0:
        raise ValueError(
            "author Mamba3 wrapper requires hidden_size * expand to be divisible by mamba_head_dim"
        )

    explicit_num_heads = getattr(megatron_config, "mamba_num_heads", None)
    implied_num_heads = d_inner // head_dim
    if explicit_num_heads is not None and explicit_num_heads != implied_num_heads:
        raise ValueError(
            "author Mamba3 wrapper does not support a custom mamba_num_heads override; "
            "leave it unset or match hidden_size * expand // mamba_head_dim"
        )

    return AuthorMamba3Config(
        d_model=hidden_size,
        d_state=state_dim,
        expand=expand,
        headdim=head_dim,
        ngroups=num_groups,
        rope_fraction=getattr(megatron_config, "cppmega_mamba3_rope_fraction", 0.5),
        dt_min=getattr(megatron_config, "cppmega_mamba3_dt_min", 0.001),
        dt_max=getattr(megatron_config, "cppmega_mamba3_dt_max", 0.1),
        dt_init_floor=getattr(megatron_config, "cppmega_mamba3_dt_init_floor", 1e-4),
        A_floor=getattr(megatron_config, "cppmega_mamba3_a_floor", 1e-4),
        is_outproj_norm=getattr(megatron_config, "cppmega_mamba3_is_outproj_norm", False),
        is_mimo=getattr(megatron_config, "cppmega_mamba3_is_mimo", False),
        mimo_rank=getattr(megatron_config, "cppmega_mamba3_mimo_rank", 4),
        chunk_size=getattr(megatron_config, "cppmega_mamba3_chunk_size", 64),
    )
