"""Fail-closed config surfaces for nanochat Engram and ngram-hash features.

These are intentionally lightweight and importable on macOS. They capture only
the non-Megatron-native feature contract so recipes can decide what remains a
custom port candidate without copying nanochat's training loop.
"""

from __future__ import annotations

from dataclasses import dataclass


def parse_layer_indices(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    values: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError(f"layer indices must be non-negative, got {value}")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)


def _parse_ngram_orders(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"ngram order must be positive, got {value}")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("at least one ngram order is required")
    return tuple(values)


@dataclass(frozen=True)
class EngramConfig:
    layer_indices: tuple[int, ...]
    ngram_orders: tuple[int, ...] = (2, 3, 4)
    bottleneck_dim: int = 0
    dropout: float = 0.0
    gated: bool = False
    gate_sqrt_compress: bool = False
    conv_kernel: int = 0
    conv_impl: str = "xla_safe"

    @classmethod
    def from_nanochat_args(
        cls,
        *,
        enabled: bool,
        layers: str,
        ngram_orders: str = "2,3,4",
        bottleneck_dim: int = 0,
        dropout: float = 0.0,
        gated: bool = False,
        gate_sqrt_compress: bool = False,
        conv_kernel: int = 0,
        conv_impl: str = "xla_safe",
    ) -> "EngramConfig | None":
        if not enabled:
            return None
        layer_indices = parse_layer_indices(layers)
        if not layer_indices:
            raise ValueError("Engram is enabled but no layer indices were provided")
        if bottleneck_dim < 0:
            raise ValueError("Engram bottleneck_dim must be >= 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Engram dropout must be in [0, 1)")
        if conv_kernel < 0:
            raise ValueError("Engram conv_kernel must be >= 0")
        if conv_impl not in {"xla_safe", "maxtext_depthwise"}:
            raise ValueError(
                f"unsupported Engram conv_impl={conv_impl!r}; expected 'xla_safe' or 'maxtext_depthwise'"
            )
        return cls(
            layer_indices=layer_indices,
            ngram_orders=_parse_ngram_orders(ngram_orders),
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            gated=gated,
            gate_sqrt_compress=gate_sqrt_compress,
            conv_kernel=conv_kernel,
            conv_impl=conv_impl,
        )


@dataclass(frozen=True)
class NgramHashConfig:
    orders: tuple[int, ...]
    heads: int
    table_size: int
    embed_dim: int
    offload: bool = False

    @classmethod
    def from_nanochat_args(
        cls,
        *,
        enabled: bool,
        orders: str = "2,3",
        heads: int = 8,
        table_size: int = 500_000,
        embed_dim: int = 16,
        offload: bool = False,
    ) -> "NgramHashConfig | None":
        if not enabled:
            return None
        if heads <= 0:
            raise ValueError("ngram hash heads must be positive")
        if table_size <= 0:
            raise ValueError("ngram hash table_size must be positive")
        if embed_dim <= 0:
            raise ValueError("ngram hash embed_dim must be positive")
        return cls(
            orders=_parse_ngram_orders(orders),
            heads=heads,
            table_size=table_size,
            embed_dim=embed_dim,
            offload=offload,
        )
