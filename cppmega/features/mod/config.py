"""Fail-closed config surfaces for nanochat MoD-family features."""

from __future__ import annotations

from dataclasses import dataclass

from cppmega.features.engram.config import parse_layer_indices


@dataclass(frozen=True)
class MoDConfig:
    layer_indices: tuple[int, ...]
    capacity: float = 0.5
    aux_loss_weight: float = 0.01
    routing: str = "topk"
    target: str = ""
    scorer: str = ""
    selector: str = ""
    schedule: str = ""
    executor: str = "auto"
    ffn_only: bool = False
    skip_first_n: int = 4
    skip_mamba: bool = True

    @classmethod
    def from_nanochat_args(
        cls,
        *,
        enabled: bool,
        layers: str,
        capacity: float = 0.5,
        aux_loss_weight: float = 0.01,
        routing: str = "topk",
        target: str = "",
        scorer: str = "",
        selector: str = "",
        schedule: str = "",
        executor: str = "auto",
        ffn_only: bool = False,
        skip_first_n: int = 4,
        skip_mamba: bool = True,
    ) -> "MoDConfig | None":
        if not enabled:
            return None
        layer_indices = parse_layer_indices(layers)
        if not layer_indices:
            raise ValueError("MoD is enabled but no layer indices were provided")
        if not 0.0 < capacity <= 1.0:
            raise ValueError("MoD capacity must be in (0, 1]")
        if aux_loss_weight < 0.0:
            raise ValueError("MoD aux_loss_weight must be >= 0")
        if routing not in {"topk", "threshold", "progressive", "gateskip"}:
            raise ValueError(f"unsupported MoD routing={routing!r}")
        if executor not in {"auto", "gather", "mask"}:
            raise ValueError(f"unsupported MoD executor={executor!r}")
        if skip_first_n < 0:
            raise ValueError("MoD skip_first_n must be >= 0")
        return cls(
            layer_indices=layer_indices,
            capacity=capacity,
            aux_loss_weight=aux_loss_weight,
            routing=routing,
            target=target,
            scorer=scorer,
            selector=selector,
            schedule=schedule,
            executor=executor,
            ffn_only=ffn_only,
            skip_first_n=skip_first_n,
            skip_mamba=skip_mamba,
        )


@dataclass(frozen=True)
class MoDAConfig:
    enabled: bool = False

    @classmethod
    def from_nanochat_args(cls, *, enabled: bool) -> "MoDAConfig | None":
        if not enabled:
            return None
        return cls(enabled=True)
