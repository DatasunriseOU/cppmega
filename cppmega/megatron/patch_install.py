"""Single entry-point for installing the cppmega Megatron feature stack.

Replaces the implicit, duplicated 4-step incantation

    apply_te_checkpoint_kwarg_patch()
    apply_dsa_indexer_fused_patch()
    apply_graph_route_attention_bias_patch()
    import cppmega.megatron.structure_dataset_patch   # import side-effects

with one call: ``install_cppmega_stack()``. It applies the patches in the
canonical order, asserts each reported success (RULE #1: fail loud, no silent
partial install), and imports the side-effect module last.

``CppMegaFeatureConfig.from_env()`` additionally parses and *validates the
combinations* of the feature-gating env flags (e.g. graph routes require
structure ingest) — invalid states raise here instead of surviving to a later
forward pass. This layer is ADDITIVE: existing env-driven call sites keep
working unchanged; the reading sites still read os.environ.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}: invalid boolean value {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}: expected integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}: expected float, got {raw!r}") from exc


@dataclass(frozen=True)
class CppMegaFeatureConfig:
    """Validated snapshot of the cppmega feature-gating env flags.

    Numeric defaults mirror the values the reading sites actually use so the
    config faithfully represents runtime behaviour (it is additive, not yet the
    source of truth).
    """

    structure_enabled: bool
    graph_routes_enabled: bool
    graph_dense_attention_bias: bool
    graph_dense_max_seq: int
    graph_max_edges: int
    graph_max_chunks: int
    domain_embedding_enabled: bool
    domain_bottleneck_dim: int
    ngram_hash_enabled: bool
    structure_components: str
    structure_bottleneck_dim: int
    graph_attention_call_weight: float
    graph_attention_type_weight: float
    graph_attention_domain_weight: float
    graph_attention_build_weight: float
    graph_attention_shell_weight: float
    graph_attention_diagnostic_weight: float
    graph_attention_cross_domain_weight: float
    dsa_graph_bias_beta: float

    @classmethod
    def from_env(cls) -> "CppMegaFeatureConfig":
        cfg = cls(
            structure_enabled=_env_bool("CPPMEGA_STRUCTURE_ENABLED"),
            graph_routes_enabled=_env_bool("CPPMEGA_GRAPH_ROUTES_ENABLED"),
            graph_dense_attention_bias=_env_bool("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", True),
            graph_dense_max_seq=_env_int("CPPMEGA_GRAPH_DENSE_MAX_SEQ", 16384),
            graph_max_edges=_env_int("CPPMEGA_GRAPH_MAX_EDGES", 256),
            graph_max_chunks=_env_int("CPPMEGA_GRAPH_MAX_CHUNKS", 256),
            domain_embedding_enabled=_env_bool("CPPMEGA_DOMAIN_EMBEDDING_ENABLED"),
            domain_bottleneck_dim=_env_int("CPPMEGA_DOMAIN_BOTTLENECK_DIM", 32),
            ngram_hash_enabled=_env_bool("CPPMEGA_NGRAM_HASH_ENABLED"),
            structure_components=os.environ.get("CPPMEGA_STRUCTURE_COMPONENTS", ""),
            structure_bottleneck_dim=_env_int("CPPMEGA_STRUCTURE_BOTTLENECK_DIM", 32),
            graph_attention_call_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_CALL_WEIGHT", 1.0),
            graph_attention_type_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_TYPE_WEIGHT", 1.0),
            graph_attention_domain_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_DOMAIN_WEIGHT", 1.0),
            graph_attention_build_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_BUILD_WEIGHT", 1.0),
            graph_attention_shell_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_SHELL_WEIGHT", 1.0),
            graph_attention_diagnostic_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_DIAGNOSTIC_WEIGHT", 1.0),
            graph_attention_cross_domain_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_CROSS_DOMAIN_WEIGHT", 1.0),
            dsa_graph_bias_beta=_env_float("CPPMEGA_DSA_GRAPH_BIAS_BETA", 1.0),
        )
        # The DENSE flag DEFAULTS on but is gated by routes at runtime
        # (graph_dense_bias_enabled() returns False when routes are off), so a
        # default dense=True with routes off is harmless — do NOT reject it. Only
        # an EXPLICIT CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS=1 with routes off is a
        # misconfiguration (the user asked for dense bias that will silently no-op).
        raw_dense = os.environ.get("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS")
        if raw_dense is not None and _env_bool("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS"):
            if not cfg.graph_routes_enabled:
                raise ValueError(
                    "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS is set but "
                    "CPPMEGA_GRAPH_ROUTES_ENABLED is not — dense graph bias is gated "
                    "off without graph routes, so this flag would silently no-op"
                )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        # RULE #1: an invalid feature combination must fail loud HERE, not survive
        # to a later forward pass where it silently runs token-only / mis-shaped.
        if self.graph_routes_enabled and not self.structure_enabled:
            raise ValueError(
                "CPPMEGA_GRAPH_ROUTES_ENABLED=1 requires CPPMEGA_STRUCTURE_ENABLED=1 "
                "(graph sidecars ride the structure ingest)"
            )
        if self.domain_embedding_enabled and not self.structure_enabled:
            raise ValueError(
                "CPPMEGA_DOMAIN_EMBEDDING_ENABLED=1 requires CPPMEGA_STRUCTURE_ENABLED=1 "
                "(domain ids arrive as structure sidecars)"
            )
        for name, value in (
            ("CPPMEGA_GRAPH_DENSE_MAX_SEQ", self.graph_dense_max_seq),
            ("CPPMEGA_GRAPH_MAX_EDGES", self.graph_max_edges),
            ("CPPMEGA_GRAPH_MAX_CHUNKS", self.graph_max_chunks),
            ("CPPMEGA_DOMAIN_BOTTLENECK_DIM", self.domain_bottleneck_dim),
            ("CPPMEGA_STRUCTURE_BOTTLENECK_DIM", self.structure_bottleneck_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


def install_cppmega_stack(config: CppMegaFeatureConfig | None = None) -> CppMegaFeatureConfig:
    """Apply the cppmega Megatron patches in canonical order, fail-loud on any gap.

    Order matters and was previously duplicated across launch scripts:
    te_checkpoint -> dsa_indexer -> graph_route_attention_bias -> structure_dataset
    (the last installs via import side-effects). Each apply_* must report True;
    anything else raises so a run never proceeds with a partially-installed stack.
    """
    if config is None:
        config = CppMegaFeatureConfig.from_env()
    else:
        config.validate()

    from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch
    from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
    from cppmega.megatron.graph_route_attention_bias_patch import (
        apply_graph_route_attention_bias_patch,
    )

    steps = (
        ("te_checkpoint_kwarg_patch", apply_te_checkpoint_kwarg_patch),
        ("dsa_indexer_fused_patch", apply_dsa_indexer_fused_patch),
        ("graph_route_attention_bias_patch", apply_graph_route_attention_bias_patch),
    )
    for name, fn in steps:
        try:
            ok = fn()
        except Exception as exc:
            raise RuntimeError(f"[cppmega-patch] {name} raised during install: {exc}") from exc
        if ok is not True:
            raise RuntimeError(f"[cppmega-patch] {name} did not report installed=True (got {ok!r})")

    # structure_dataset_patch installs GPTDataset/get_batch/model.forward patches at
    # import time; do it explicitly last so ordering is deterministic, not implicit.
    try:
        importlib.import_module("cppmega.megatron.structure_dataset_patch")
    except Exception as exc:
        raise RuntimeError(f"[cppmega-patch] structure_dataset_patch import failed: {exc}") from exc

    return config


__all__ = ["CppMegaFeatureConfig", "install_cppmega_stack"]
