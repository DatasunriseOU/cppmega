from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from cppmega.megatron.graph_objective_loss import (
    GRAPH_BIAS_BETA_ENV,
    GRAPH_BIAS_BETA_LEGACY_ENVS,
    GraphAuxiliaryLossConfig,
    compose_dsa_indexer_total_loss,
    graph_bias_beta_binding,
    graph_auxiliary_loss,
    require_active_dsa_graph_objective,
    resolve_graph_bias_beta,
    validate_runtime_graph_contract,
)
from cppmega.megatron.graph_recipe import (
    stage1_graph_recipe_binding,
    stage1_graph_recipe_payload,
)


def _included_graph_contract() -> dict[str, object]:
    return {
        **stage1_graph_recipe_payload(),
        "recipe": stage1_graph_recipe_binding(),
        "eligible_samples": 1,
        "positive_edges": 1,
        "included_in_total_loss": True,
    }


def test_weighted_graph_losses_enter_total_and_backpropagate() -> None:
    scores = torch.zeros((1, 4, 4), dtype=torch.float32, requires_grad=True)
    targets = torch.zeros_like(scores)
    targets[0, 2, 0] = 1.0
    pair_mask = torch.tril(torch.ones_like(scores))
    config = GraphAuxiliaryLossConfig(
        global_weight=0.5,
        indexer_weight=0.125,
        layer_weight=0.25,
        bce_weight=1.0,
        coverage_weight=0.25,
        topk=1,
        pos_weight=2.0,
        margin=1.0,
    )

    graph_loss, components = graph_auxiliary_loss(
        scores, targets, pair_mask=pair_mask, config=config
    )
    lm_loss = scores.new_tensor(3.0)
    total = lm_loss + graph_loss
    total.backward()

    assert graph_loss.item() > 0.0
    assert graph_loss.item() == pytest.approx(
        0.125
        * 0.5
        * 0.25
        * (components["bce"].item() + 0.25 * components["coverage"].item())
    )
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad).item() > 0


def test_indexer_weight_scales_graph_loss_exactly_once() -> None:
    scores = torch.tensor(
        [[[0.25, -0.5], [0.75, -1.0]]], dtype=torch.float32
    )
    targets = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32
    )
    pair_mask = torch.tensor(
        [[[True, False], [True, True]]], dtype=torch.bool
    )
    common = {
        "global_weight": 0.5,
        "layer_weight": 0.25,
        "bce_weight": 1.0,
        "coverage_weight": 0.25,
        "topk": 1,
        "pos_weight": 2.0,
        "margin": 1.0,
    }

    unit_loss, _ = graph_auxiliary_loss(
        scores,
        targets,
        pair_mask=pair_mask,
        config=GraphAuxiliaryLossConfig(indexer_weight=1.0, **common),
    )
    milli_loss, _ = graph_auxiliary_loss(
        scores,
        targets,
        pair_mask=pair_mask,
        config=GraphAuxiliaryLossConfig(indexer_weight=0.001, **common),
    )

    assert torch.equal(milli_loss, unit_loss * 0.001)


def test_indexer_and_graph_losses_compose_into_one_total_loss_scalar() -> None:
    indexer_parameter = torch.tensor(2.0, requires_grad=True)
    graph_parameter = torch.tensor(3.0, requires_grad=True)
    indexer_loss = indexer_parameter.square()
    graph_loss = graph_parameter.square()

    total_loss = compose_dsa_indexer_total_loss(indexer_loss, graph_loss)
    total_loss.backward()

    assert total_loss.ndim == 0
    assert total_loss.item() == pytest.approx(13.0)
    assert indexer_parameter.grad is not None
    assert graph_parameter.grad is not None
    assert indexer_parameter.grad.item() == pytest.approx(4.0)
    assert graph_parameter.grad.item() == pytest.approx(6.0)


def test_graph_loss_ignores_graph_ineligible_batch_without_fabricating_edges() -> None:
    scores = torch.randn((2, 4, 4), requires_grad=True)
    targets = torch.zeros_like(scores)
    pair_mask = torch.tril(torch.ones_like(scores))
    config = GraphAuxiliaryLossConfig(
        global_weight=0.5,
        bce_weight=1.0,
        coverage_weight=0.25,
        topk=1,
    )

    graph_loss, components = graph_auxiliary_loss(
        scores, targets, pair_mask=pair_mask, config=config
    )

    assert graph_loss.item() == 0.0
    assert components["positive_edges"].item() == 0


def test_graph_loss_sanitizes_non_finite_masked_scores_and_gradients() -> None:
    scores = torch.tensor(
        [[[0.0, float("inf")], [0.25, float("nan")]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0]]],
        dtype=torch.float32,
    )
    pair_mask = torch.tensor(
        [[[True, False], [True, False]]],
        dtype=torch.bool,
    )
    config = GraphAuxiliaryLossConfig(
        global_weight=1.0,
        indexer_weight=1.0,
        layer_weight=1.0,
        bce_weight=1.0,
        coverage_weight=1.0,
        topk=2,
        pos_weight=1.0,
        margin=1.0,
    )

    graph_loss, _ = graph_auxiliary_loss(
        scores,
        targets,
        pair_mask=pair_mask,
        config=config,
    )
    graph_loss.backward()

    assert torch.isfinite(graph_loss)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad).item() > 0


def test_graph_loss_rejects_positive_targets_at_non_finite_scores() -> None:
    scores = torch.tensor(
        [[[0.0, float("inf"), -1.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    pair_mask = torch.ones_like(scores, dtype=torch.bool)
    config = GraphAuxiliaryLossConfig(
        global_weight=1.0,
        indexer_weight=1.0,
        layer_weight=1.0,
        bce_weight=1.0,
        coverage_weight=1.0,
        topk=1,
        pos_weight=1.0,
        margin=3.0,
    )

    with pytest.raises(ValueError, match="positive graph target.*non-finite"):
        graph_auxiliary_loss(
            scores,
            targets,
            pair_mask=pair_mask,
            config=config,
        )


def test_graph_loss_empty_batch_returns_finite_connected_zero() -> None:
    scores = torch.tensor(
        [[[float("inf"), float("-inf")], [float("nan"), 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.zeros_like(scores)
    pair_mask = torch.ones_like(scores, dtype=torch.bool)
    config = GraphAuxiliaryLossConfig(
        global_weight=1.0,
        indexer_weight=1.0,
        layer_weight=1.0,
        bce_weight=1.0,
        coverage_weight=1.0,
        topk=1,
        pos_weight=1.0,
        margin=1.0,
    )

    graph_loss, _ = graph_auxiliary_loss(
        scores,
        targets,
        pair_mask=pair_mask,
        config=config,
    )
    graph_loss.backward()

    assert graph_loss.item() == 0.0
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad).item() == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_weight", 0.0),
        ("indexer_weight", 0.0),
        ("layer_weight", 0.0),
        ("bce_weight", 0.0),
        ("coverage_weight", -1.0),
        ("topk", 0),
    ],
)
def test_graph_loss_config_fails_closed_on_disabled_or_invalid_components(
    field: str, value: float
) -> None:
    kwargs = {
        "global_weight": 0.5,
        "bce_weight": 1.0,
        "coverage_weight": 0.25,
        "topk": 1,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        GraphAuxiliaryLossConfig(**kwargs)


def test_graph_enabled_model_requires_active_dense_dsa_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")

    with pytest.raises(ValueError, match="dsa_indexer_loss_coeff"):
        require_active_dsa_graph_objective(
            SimpleNamespace(
                dsa_indexer_loss_coeff=0.0,
                dsa_indexer_use_sparse_loss=False,
            )
        )
    with pytest.raises(ValueError, match="dsa_indexer_use_sparse_loss"):
        require_active_dsa_graph_objective(
            SimpleNamespace(
                dsa_indexer_loss_coeff=0.001,
                dsa_indexer_use_sparse_loss=True,
            )
        )

    require_active_dsa_graph_objective(
        SimpleNamespace(
            dsa_indexer_loss_coeff=0.001,
            dsa_indexer_use_sparse_loss=False,
        )
    )

    monkeypatch.setenv("CPPMEGA_DSA_INDEXER_LOSS_COEFF", "0.002")
    with pytest.raises(ValueError, match="differs from the graph indexer coefficient"):
        require_active_dsa_graph_objective(
            SimpleNamespace(
                dsa_indexer_loss_coeff=0.001,
                dsa_indexer_use_sparse_loss=False,
            )
        )

    with pytest.raises(ValueError, match="finite positive"):
        require_active_dsa_graph_objective(
            SimpleNamespace(
                dsa_indexer_loss_coeff=float("nan"),
                dsa_indexer_use_sparse_loss=False,
            )
        )


def test_graph_objective_fails_when_routes_or_structure_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        dsa_indexer_loss_coeff=0.001,
        dsa_indexer_use_sparse_loss=False,
    )
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")
    with pytest.raises(ValueError, match="GRAPH_ROUTES_ENABLED.*disabled"):
        require_active_dsa_graph_objective(config)

    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    with pytest.raises(ValueError, match="STRUCTURE_ENABLED.*disabled"):
        require_active_dsa_graph_objective(config)


def test_dense_graph_routes_do_not_require_a_dsa_indexer_coefficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.delenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", raising=False)

    require_active_dsa_graph_objective(
        SimpleNamespace(dsa_indexer_loss_coeff=None), required=False
    )


def test_graph_runtime_config_rejects_non_finite_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_WEIGHT", "nan")

    with pytest.raises(ValueError, match="global_weight"):
        GraphAuxiliaryLossConfig.from_env()


def test_included_graph_contract_fails_closed_when_auxiliary_is_disabled() -> None:
    environment = {
        "CPPMEGA_STRUCTURE_ENABLED": "1",
        "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
        "CPPMEGA_DSA_GRAPH_AUX_ENABLED": "0",
    }

    with pytest.raises(ValueError, match="included_in_total_loss.*auxiliary"):
        validate_runtime_graph_contract(
            _included_graph_contract(),
            environment=environment,
        )


def test_runtime_graph_contract_rejects_selector_beta_drift() -> None:
    contract = _included_graph_contract()
    contract["bias_beta"] = "1"
    environment = {
        "CPPMEGA_STRUCTURE_ENABLED": "1",
        "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
        "CPPMEGA_DSA_GRAPH_AUX_ENABLED": "1",
        "CPPMEGA_DSA_GRAPH_BIAS_BETA": "0",
    }

    with pytest.raises(ValueError, match="bias beta"):
        validate_runtime_graph_contract(contract, environment=environment)


def test_graph_config_rejects_dsa_dense_beta_drift() -> None:
    environment = {
        "CPPMEGA_DSA_GRAPH_BIAS_BETA": "2",
        "CPPMEGA_GRAPH_ATTENTION_BIAS_BETA": "3",
    }

    with pytest.raises(ValueError, match="beta.*differ"):
        GraphAuxiliaryLossConfig.from_env(environment)


def test_graph_config_binds_canonical_beta_and_equal_legacy_aliases() -> None:
    environment = {
        GRAPH_BIAS_BETA_ENV: "2",
        GRAPH_BIAS_BETA_LEGACY_ENVS[0]: "2.0",
        GRAPH_BIAS_BETA_LEGACY_ENVS[1]: "2e0",
    }

    assert resolve_graph_bias_beta(environment) == 2.0
    config = GraphAuxiliaryLossConfig.from_env(environment)
    assert config.bias_beta == 2.0
    assert graph_bias_beta_binding(config.bias_beta) == {
        "canonical_env": GRAPH_BIAS_BETA_ENV,
        "legacy_envs": list(GRAPH_BIAS_BETA_LEGACY_ENVS),
        "value": "2",
    }
