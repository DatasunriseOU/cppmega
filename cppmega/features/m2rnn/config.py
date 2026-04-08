"""Config helpers for the minimal cppmega M2RNN seam."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CppMegaM2RNNConfig:
    d_model: int
    k_head_dim: int = 64
    v_head_dim: int = 16
    conv_kernel: int = 4
    gradient_clipping: float = 1.0
    use_residual: bool = True
    A_init_min: float = 0.0
    A_init_max: float = 16.0
    dt_init_min: float = 1e-3
    dt_init_max: float = 0.1
    dt_init_floor: float = 1e-4
    use_xma: bool = False


def build_cppmega_m2rnn_config(megatron_config, *, d_model: int | None = None) -> CppMegaM2RNNConfig:
    hidden_size = getattr(megatron_config, "hidden_size", None) if d_model is None else d_model
    if hidden_size is None:
        raise ValueError("hidden_size or d_model must be provided for cppmega M2RNN config")

    return CppMegaM2RNNConfig(
        d_model=hidden_size,
        k_head_dim=getattr(megatron_config, "m2rnn_k_head_dim", 64),
        v_head_dim=getattr(megatron_config, "m2rnn_v_head_dim", 16),
        conv_kernel=getattr(megatron_config, "m2rnn_conv_kernel", 4),
        gradient_clipping=getattr(megatron_config, "m2rnn_gradient_clipping", 1.0),
        use_residual=getattr(megatron_config, "m2rnn_use_residual", True),
        A_init_min=getattr(megatron_config, "m2rnn_A_init_min", 0.0),
        A_init_max=getattr(megatron_config, "m2rnn_A_init_max", 16.0),
        dt_init_min=getattr(megatron_config, "m2rnn_dt_init_min", 1e-3),
        dt_init_max=getattr(megatron_config, "m2rnn_dt_init_max", 0.1),
        dt_init_floor=getattr(megatron_config, "m2rnn_dt_init_floor", 1e-4),
        use_xma=getattr(megatron_config, "m2rnn_use_xma", False),
    )
