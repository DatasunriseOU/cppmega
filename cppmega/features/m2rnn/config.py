"""Config helpers for the minimal cppmega M2RNN seam."""

from __future__ import annotations

import os
from dataclasses import dataclass


_DEFAULT_RUNTIME_KERNEL = "triton"
_DEFAULT_SAVE_HNEW = False
_DEFAULT_BWD_CHUNK_SIZE = 64
_DEFAULT_FWD_AUTOTUNE = False
_DEFAULT_FWD_NUM_WARPS = 4
_DEFAULT_FWD_NUM_STAGES = 3
_DEFAULT_BROADCAST_VIEWS = True
_DEFAULT_BWD_REDUCE_BROADCAST_QK = False


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
    runtime_kernel: str = _DEFAULT_RUNTIME_KERNEL
    runtime_save_hnew: bool = _DEFAULT_SAVE_HNEW
    runtime_bwd_chunk_size: int = _DEFAULT_BWD_CHUNK_SIZE
    runtime_fwd_autotune: bool = _DEFAULT_FWD_AUTOTUNE
    runtime_fwd_num_warps: int = _DEFAULT_FWD_NUM_WARPS
    runtime_fwd_num_stages: int = _DEFAULT_FWD_NUM_STAGES
    runtime_broadcast_views: bool = _DEFAULT_BROADCAST_VIEWS
    runtime_bwd_reduce_broadcast_qk: bool = _DEFAULT_BWD_REDUCE_BROADCAST_QK


def _raw_config_or_env(megatron_config, attr: str, env_name: str, default):
    if hasattr(megatron_config, attr):
        return getattr(megatron_config, attr)
    return os.environ.get(env_name, default)


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value == "1"
    return bool(value)


def _as_positive_int(value, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _as_int_choice(value, default: int, choices: set[int]) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed in choices else default


def _as_kernel(value) -> str:
    value = str(value)
    return value if value in {"triton", "torch"} else _DEFAULT_RUNTIME_KERNEL


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
        runtime_kernel=_as_kernel(
            _raw_config_or_env(
                megatron_config,
                "m2rnn_kernel",
                "CPPMEGA_M2RNN_KERNEL",
                _DEFAULT_RUNTIME_KERNEL,
            )
        ),
        runtime_save_hnew=_as_bool(
            _raw_config_or_env(
                megatron_config,
                "m2rnn_save_hnew",
                "CPPMEGA_M2RNN_SAVE_HNEW",
                _DEFAULT_SAVE_HNEW,
            ),
            _DEFAULT_SAVE_HNEW,
        ),
        runtime_bwd_chunk_size=_as_positive_int(
            _raw_config_or_env(
                megatron_config,
                "m2rnn_bwd_chunk_size",
                "CPPMEGA_M2RNN_BWD_CHUNK_SIZE",
                _DEFAULT_BWD_CHUNK_SIZE,
            ),
            _DEFAULT_BWD_CHUNK_SIZE,
        ),
        runtime_fwd_autotune=_as_bool(
            _raw_config_or_env(
                megatron_config,
                "m2rnn_fwd_autotune",
                "CPPMEGA_M2RNN_FWD_AUTOTUNE",
                _DEFAULT_FWD_AUTOTUNE,
            ),
            _DEFAULT_FWD_AUTOTUNE,
        ),
        runtime_fwd_num_warps=_as_int_choice(
            _raw_config_or_env(
                megatron_config,
                "m2rnn_fwd_num_warps",
                "CPPMEGA_M2RNN_FWD_NUM_WARPS",
                _DEFAULT_FWD_NUM_WARPS,
            ),
            _DEFAULT_FWD_NUM_WARPS,
            {1, 2, 4, 8, 16},
        ),
        runtime_fwd_num_stages=_as_int_choice(
            _raw_config_or_env(
                megatron_config,
                "m2rnn_fwd_num_stages",
                "CPPMEGA_M2RNN_FWD_NUM_STAGES",
                _DEFAULT_FWD_NUM_STAGES,
            ),
            _DEFAULT_FWD_NUM_STAGES,
            {1, 2, 3, 4},
        ),
        runtime_broadcast_views=_as_bool(
            _raw_config_or_env(
                megatron_config,
                "m2rnn_broadcast_views",
                "CPPMEGA_M2RNN_BROADCAST_VIEWS",
                _DEFAULT_BROADCAST_VIEWS,
            ),
            _DEFAULT_BROADCAST_VIEWS,
        ),
        runtime_bwd_reduce_broadcast_qk=_as_bool(
            _raw_config_or_env(
                megatron_config,
                "m2rnn_bwd_reduce_broadcast_qk",
                "CPPMEGA_M2RNN_BWD_REDUCE_BROADCAST_QK",
                _DEFAULT_BWD_REDUCE_BROADCAST_QK,
            ),
            _DEFAULT_BWD_REDUCE_BROADCAST_QK,
        ),
    )
