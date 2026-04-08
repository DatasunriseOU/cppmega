"""Fail-closed config surface for nanochat mHC features.

This captures only the feature contract that currently has no verified
Megatron-native equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

from cppmega.features.engram.config import parse_layer_indices


@dataclass(frozen=True)
class MHCConfig:
    layer_indices: tuple[int, ...]
    n_streams: int = 4
    sinkhorn_iters: int = 5
    temperature: float = 1.0
    epsilon: float = 1e-6
    blend_alpha: float = 1.0
    dynamic: bool = False
    dynamic_mode: str = "maxtext"
    fused_ops: bool = False
    recompute_group_size: int = 0

    @classmethod
    def from_nanochat_args(
        cls,
        *,
        enabled: bool,
        layers: str,
        n_streams: int = 4,
        sinkhorn_iters: int = 5,
        temperature: float = 1.0,
        epsilon: float = 1e-6,
        blend_alpha: float = 1.0,
        dynamic: bool = False,
        dynamic_mode: str = "maxtext",
        fused_ops: bool = False,
        recompute_group_size: int = 0,
    ) -> "MHCConfig | None":
        if not enabled:
            return None
        layer_indices = parse_layer_indices(layers)
        if not layer_indices:
            raise ValueError("mHC is enabled but no layer indices were provided")
        if n_streams <= 1:
            raise ValueError("mHC n_streams must be > 1")
        if sinkhorn_iters <= 0:
            raise ValueError("mHC sinkhorn_iters must be positive")
        if temperature <= 0.0:
            raise ValueError("mHC temperature must be positive")
        if epsilon <= 0.0:
            raise ValueError("mHC epsilon must be positive")
        if blend_alpha < 0.0:
            raise ValueError("mHC blend_alpha must be >= 0")
        if dynamic_mode not in {"maxtext", "pooled"}:
            raise ValueError(
                f"unsupported mHC dynamic_mode={dynamic_mode!r}; expected 'maxtext' or 'pooled'"
            )
        if recompute_group_size < -1 or recompute_group_size == 0:
            # 0 is allowed as nanochat auto-mode, preserve it.
            pass
        return cls(
            layer_indices=layer_indices,
            n_streams=n_streams,
            sinkhorn_iters=sinkhorn_iters,
            temperature=temperature,
            epsilon=epsilon,
            blend_alpha=blend_alpha,
            dynamic=dynamic,
            dynamic_mode=dynamic_mode,
            fused_ops=fused_ops,
            recompute_group_size=recompute_group_size,
        )
