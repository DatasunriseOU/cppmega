from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from cppmega.megatron.graph_objective_loss import (
    GraphAuxiliaryLossConfig,
    graph_auxiliary_loss,
    require_active_dsa_graph_objective,
)


def test_weighted_graph_losses_enter_total_and_backpropagate() -> None:
    scores = torch.zeros((1, 4, 4), dtype=torch.float32, requires_grad=True)
    targets = torch.zeros_like(scores)
    targets[0, 2, 0] = 1.0
    pair_mask = torch.tril(torch.ones_like(scores))
    config = GraphAuxiliaryLossConfig(
        global_weight=0.5,
        bce_weight=1.0,
        coverage_weight=0.25,
        topk=1,
        pos_weight=2.0,
        margin=1.0,
    )

    graph_loss, components = graph_auxiliary_loss(
        scores, targets, pair_mask=pair_mask, config=config
    )
    lm_loss = scores.sum() * 0.0 + 3.0
    total = lm_loss + graph_loss
    total.backward()

    assert graph_loss.item() > 0.0
    assert graph_loss.item() == pytest.approx(
        0.5 * (components["bce"].item() + 0.25 * components["coverage"].item())
    )
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad).item() > 0


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_weight", 0.0),
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

    with pytest.raises(ValueError, match="finite positive"):
        require_active_dsa_graph_objective(
            SimpleNamespace(
                dsa_indexer_loss_coeff=float("nan"),
                dsa_indexer_use_sparse_loss=False,
            )
        )


def test_graph_runtime_config_rejects_non_finite_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_WEIGHT", "nan")

    with pytest.raises(ValueError, match="global_weight"):
        GraphAuxiliaryLossConfig.from_env()
