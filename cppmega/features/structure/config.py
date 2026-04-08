"""Fail-closed config surface for nanochat structure-aware training features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureConfig:
    components: str = "core"
    max_ast_depth: int = 20
    max_sibling_index: int = 10
    num_node_types: int = 64
    bottleneck_dim: int = 64
    relation_bias_enabled: bool = False
    tree_ffn_enabled: bool = False
    tree_ffn_steps: int = 3
    tree_ffn_dropout: float = 0.0
    platform_embed_enabled: bool = False

    @classmethod
    def from_nanochat_args(
        cls,
        *,
        enabled: bool,
        components: str = "core",
        max_ast_depth: int = 20,
        max_sibling_index: int = 10,
        num_node_types: int = 64,
        bottleneck_dim: int = 64,
        relation_bias_enabled: bool = False,
        tree_ffn_enabled: bool = False,
        tree_ffn_steps: int = 3,
        tree_ffn_dropout: float = 0.0,
        platform_embed_enabled: bool = False,
    ) -> "StructureConfig | None":
        if not enabled:
            return None
        if max_ast_depth <= 0:
            raise ValueError("structure max_ast_depth must be positive")
        if max_sibling_index < 0:
            raise ValueError("structure max_sibling_index must be >= 0")
        if num_node_types <= 0:
            raise ValueError("structure num_node_types must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("structure bottleneck_dim must be positive")
        if tree_ffn_steps <= 0:
            raise ValueError("structure tree_ffn_steps must be positive")
        if not 0.0 <= tree_ffn_dropout < 1.0:
            raise ValueError("structure tree_ffn_dropout must be in [0, 1)")
        return cls(
            components=components,
            max_ast_depth=max_ast_depth,
            max_sibling_index=max_sibling_index,
            num_node_types=num_node_types,
            bottleneck_dim=bottleneck_dim,
            relation_bias_enabled=relation_bias_enabled,
            tree_ffn_enabled=tree_ffn_enabled,
            tree_ffn_steps=tree_ffn_steps,
            tree_ffn_dropout=tree_ffn_dropout,
            platform_embed_enabled=platform_embed_enabled,
        )
