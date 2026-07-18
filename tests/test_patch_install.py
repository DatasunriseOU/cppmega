from __future__ import annotations

import pytest

from cppmega.megatron.patch_install import CppMegaFeatureConfig, install_cppmega_stack

_FEATURE_FLAGS = (
    "CPPMEGA_STRUCTURE_ENABLED",
    "CPPMEGA_GRAPH_ROUTES_ENABLED",
    "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS",
    "CPPMEGA_DOMAIN_EMBEDDING_ENABLED",
    "CPPMEGA_NGRAM_HASH_ENABLED",
    "CPPMEGA_GRAPH_DENSE_MAX_SEQ",
)


@pytest.fixture(autouse=True)
def _clean_feature_env(monkeypatch):
    for flag in _FEATURE_FLAGS:
        monkeypatch.delenv(flag, raising=False)


def test_defaults_all_off_is_valid():
    cfg = CppMegaFeatureConfig.from_env()
    assert not cfg.structure_enabled
    assert not cfg.graph_routes_enabled
    assert cfg.graph_dense_max_seq == 16384  # mirrors the runtime patch default
    assert cfg.graph_max_edges == 256
    assert cfg.domain_bottleneck_dim == 32


def test_default_dense_on_with_routes_off_does_not_raise():
    # DENSE defaults on but is gated by routes at runtime; the unset-default case
    # must NOT be rejected (regression for the naive `dense and not routes` check).
    cfg = CppMegaFeatureConfig.from_env()
    assert cfg.graph_dense_attention_bias is True
    assert not cfg.graph_routes_enabled


def test_valid_full_stack():
    cfg = CppMegaFeatureConfig.from_env(
        {
            "CPPMEGA_STRUCTURE_ENABLED": "1",
            "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
            "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS": "1",
            "CPPMEGA_GRAPH_MAX_EDGES": "7",
            "CPPMEGA_GRAPH_MAX_CHUNKS": "5",
        }
    )
    assert cfg.structure_enabled and cfg.graph_routes_enabled and cfg.graph_dense_attention_bias


def test_graph_routes_require_structure(monkeypatch):
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    with pytest.raises(ValueError, match="requires CPPMEGA_STRUCTURE_ENABLED"):
        CppMegaFeatureConfig.from_env()


def test_graph_routes_require_csr_derived_capacities():
    with pytest.raises(ValueError, match="CPPMEGA_GRAPH_MAX_EDGES is required"):
        CppMegaFeatureConfig.from_env(
            {
                "CPPMEGA_STRUCTURE_ENABLED": "1",
                "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
            }
        )


def test_explicit_dense_requires_routes(monkeypatch):
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", "1")
    with pytest.raises(ValueError, match="CPPMEGA_GRAPH_ROUTES_ENABLED is not"):
        CppMegaFeatureConfig.from_env()


def test_domain_requires_structure(monkeypatch):
    monkeypatch.setenv("CPPMEGA_DOMAIN_EMBEDDING_ENABLED", "1")
    with pytest.raises(ValueError, match="requires CPPMEGA_STRUCTURE_ENABLED"):
        CppMegaFeatureConfig.from_env()


def test_bad_boolean_value_raises(monkeypatch):
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "maybe")
    with pytest.raises(ValueError, match="invalid boolean"):
        CppMegaFeatureConfig.from_env()


def test_zero_graph_max_edges_raises(monkeypatch):
    monkeypatch.setenv("CPPMEGA_GRAPH_MAX_EDGES", "0")
    with pytest.raises(ValueError, match="GRAPH_MAX_EDGES must be positive"):
        CppMegaFeatureConfig.from_env()


def test_install_applies_in_canonical_order(monkeypatch):
    import cppmega.megatron.te_checkpoint_kwarg_patch as te
    import cppmega.megatron.dsa_indexer_fused_patch as dsa
    import cppmega.megatron.graph_route_attention_bias_patch as gr

    calls: list[str] = []
    monkeypatch.setattr(te, "apply_te_checkpoint_kwarg_patch", lambda: calls.append("te") or True)
    monkeypatch.setattr(dsa, "apply_dsa_indexer_fused_patch", lambda: calls.append("dsa") or True)
    monkeypatch.setattr(gr, "apply_graph_route_attention_bias_patch", lambda: calls.append("gr") or True)
    config = CppMegaFeatureConfig.from_env(
        {
            "CPPMEGA_STRUCTURE_ENABLED": "1",
            "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
            "CPPMEGA_GRAPH_MAX_EDGES": "7",
            "CPPMEGA_GRAPH_MAX_CHUNKS": "5",
        }
    )

    cfg = install_cppmega_stack(config)
    assert calls == ["te", "dsa", "gr"]  # canonical order, structure imported last
    assert cfg.structure_enabled and cfg.graph_routes_enabled


def test_install_raises_if_patch_reports_not_installed(monkeypatch):
    import cppmega.megatron.te_checkpoint_kwarg_patch as te
    import cppmega.megatron.dsa_indexer_fused_patch as dsa

    monkeypatch.setattr(te, "apply_te_checkpoint_kwarg_patch", lambda: True)
    monkeypatch.setattr(dsa, "apply_dsa_indexer_fused_patch", lambda: False)
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    with pytest.raises(RuntimeError, match="did not report installed"):
        install_cppmega_stack()


def test_install_raises_if_patch_raises(monkeypatch):
    import cppmega.megatron.te_checkpoint_kwarg_patch as te

    def boom():
        raise RuntimeError("megatron missing")

    monkeypatch.setattr(te, "apply_te_checkpoint_kwarg_patch", boom)
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    with pytest.raises(RuntimeError, match="raised during install"):
        install_cppmega_stack()
