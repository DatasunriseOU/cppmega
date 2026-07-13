"""Differentiable graph supervision for production DSA indexer scores."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GraphAuxiliaryLossConfig:
    global_weight: float
    bce_weight: float
    coverage_weight: float
    topk: int
    pos_weight: float = 1.0
    margin: float = 1.0
    relations: tuple[str, ...] = ("call", "type")

    def __post_init__(self) -> None:
        for name in ("global_weight", "bce_weight", "coverage_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.topk < 1:
            raise ValueError("topk must be >= 1")
        if not math.isfinite(float(self.pos_weight)) or self.pos_weight <= 0.0:
            raise ValueError("pos_weight must be > 0")
        if not math.isfinite(float(self.margin)) or self.margin < 0.0:
            raise ValueError("margin must be >= 0")
        if not self.relations or any(not relation for relation in self.relations):
            raise ValueError("relations must contain at least one non-empty name")
        if len(set(self.relations)) != len(self.relations):
            raise ValueError("relations must not contain duplicates")

    @classmethod
    def from_env(cls) -> "GraphAuxiliaryLossConfig":
        relations = tuple(
            item.strip()
            for item in os.environ.get(
                "CPPMEGA_DSA_GRAPH_AUX_RELATIONS", "call,type"
            ).split(",")
            if item.strip()
        )
        try:
            return cls(
                global_weight=float(
                    os.environ.get("CPPMEGA_DSA_GRAPH_AUX_WEIGHT", "1.0")
                ),
                bce_weight=float(
                    os.environ.get("CPPMEGA_DSA_GRAPH_BCE_WEIGHT", "0.10")
                ),
                coverage_weight=float(
                    os.environ.get("CPPMEGA_DSA_GRAPH_COVERAGE_WEIGHT", "0.05")
                ),
                topk=int(os.environ.get("CPPMEGA_DSA_GRAPH_AUX_TOPK", "8")),
                pos_weight=float(os.environ.get("CPPMEGA_DSA_GRAPH_POS_WEIGHT", "1.0")),
                margin=float(os.environ.get("CPPMEGA_DSA_GRAPH_MARGIN", "1.0")),
                relations=relations,
            )
        except ValueError as exc:
            raise ValueError(f"invalid DSA graph auxiliary environment: {exc}") from exc


def validate_runtime_graph_contract(graph_contract: Mapping[str, object]) -> None:
    """Require runtime graph-loss knobs to exactly match the data receipt."""

    config = GraphAuxiliaryLossConfig.from_env()
    expected_relations = tuple(graph_contract.get("relations", ()))
    if config.relations != expected_relations:
        raise ValueError(
            "graph auxiliary runtime relations differ from contract: "
            f"runtime={config.relations}, contract={expected_relations}"
        )
    comparisons = {
        "global_weight": config.global_weight,
        "bce_weight": config.bce_weight,
        "coverage_weight": config.coverage_weight,
        "pos_weight": config.pos_weight,
        "margin": config.margin,
    }
    for field, runtime_value in comparisons.items():
        raw_contract = graph_contract.get(field)
        if not isinstance(raw_contract, str):
            raise ValueError(f"graph_auxiliary.{field} contract value must be exact")
        if Fraction(str(runtime_value)) != Fraction(raw_contract):
            raise ValueError(
                f"graph auxiliary runtime {field}={runtime_value} differs from "
                f"contract={raw_contract}"
            )
    if config.topk != graph_contract.get("topk"):
        raise ValueError(
            f"graph auxiliary runtime topk={config.topk} differs from "
            f"contract={graph_contract.get('topk')!r}"
        )


def require_active_dsa_graph_objective(transformer_config: object) -> None:
    """Fail before model construction if graph loss cannot reach total loss."""

    if os.environ.get("CPPMEGA_GRAPH_ROUTES_ENABLED", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    coefficient = getattr(transformer_config, "dsa_indexer_loss_coeff", None)
    coefficient_value = float(coefficient) if coefficient is not None else None
    if (
        coefficient_value is None
        or not math.isfinite(coefficient_value)
        or coefficient_value <= 0.0
    ):
        raise ValueError(
            "CPPMEGA_GRAPH_ROUTES_ENABLED=1 requires a finite positive "
            "TransformerConfig.dsa_indexer_loss_coeff so the weighted graph "
            "objective reaches DSAIndexerLossAutoScaler"
        )
    if bool(getattr(transformer_config, "dsa_indexer_use_sparse_loss", False)):
        raise ValueError(
            "CPPMEGA_GRAPH_ROUTES_ENABLED=1 requires "
            "dsa_indexer_use_sparse_loss=False because graph BCE needs full "
            "indexer scores"
        )


def _validate_inputs(
    indexer_scores: torch.Tensor,
    edge_targets: torch.Tensor,
    pair_mask: torch.Tensor,
) -> tuple[int, int, int]:
    if indexer_scores.ndim != 3:
        raise ValueError(
            "indexer_scores must be (B,Q,K), got " f"{tuple(indexer_scores.shape)}"
        )
    if edge_targets.shape != indexer_scores.shape:
        raise ValueError(
            f"edge_targets shape {tuple(edge_targets.shape)} must match "
            f"indexer_scores {tuple(indexer_scores.shape)}"
        )
    if pair_mask.shape != indexer_scores.shape:
        raise ValueError(
            f"pair_mask shape {tuple(pair_mask.shape)} must match "
            f"indexer_scores {tuple(indexer_scores.shape)}"
        )
    if not torch.all((edge_targets == 0) | (edge_targets == 1)):
        raise ValueError("edge_targets must contain only 0/1 values")
    if torch.any((edge_targets > 0) & (pair_mask <= 0)):
        raise ValueError("edge_targets contains a positive edge outside pair_mask")
    return tuple(int(value) for value in indexer_scores.shape)


def graph_auxiliary_loss(
    indexer_scores: torch.Tensor,
    edge_targets: torch.Tensor,
    *,
    pair_mask: torch.Tensor,
    config: GraphAuxiliaryLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return weighted graph loss and exact differentiable components.

    Samples without positive graph edges are excluded rather than assigned
    fabricated negatives. A fully graph-ineligible batch returns a connected
    zero scalar; dataset-level eligibility is enforced by the objective receipt.
    """

    _batch, _queries, keys = _validate_inputs(indexer_scores, edge_targets, pair_mask)
    scores = indexer_scores.float()
    targets = edge_targets.to(device=scores.device, dtype=torch.float32)
    valid_pairs = pair_mask.to(device=scores.device, dtype=torch.bool)
    positive_edges = targets.sum()
    if int(positive_edges.detach().item()) == 0:
        zero = scores.sum() * 0.0
        return zero, {
            "bce": zero,
            "coverage": zero,
            "positive_edges": positive_edges,
        }

    eligible_samples = targets.sum(dim=(1, 2)) > 0
    valid_pairs = valid_pairs & eligible_samples[:, None, None]
    element_bce = F.binary_cross_entropy_with_logits(
        scores,
        targets,
        reduction="none",
        pos_weight=torch.tensor(
            config.pos_weight, device=scores.device, dtype=scores.dtype
        ),
    )
    bce = element_bce[valid_pairs].mean()

    if config.topk >= keys:
        coverage = scores.sum() * 0.0
    else:
        masked_scores = scores.masked_fill(~valid_pairs, float("-inf"))
        boundary = torch.topk(masked_scores, k=config.topk + 1, dim=-1).values[..., -1:]
        finite_boundary = torch.isfinite(boundary)
        deficits = torch.relu(boundary + config.margin - scores)
        penalties = deficits * targets * finite_boundary.to(scores.dtype)
        coverage = penalties.sum() / positive_edges.clamp_min(1.0)

    total = config.global_weight * (
        config.bce_weight * bce + config.coverage_weight * coverage
    )
    return total, {
        "bce": bce,
        "coverage": coverage,
        "positive_edges": positive_edges,
    }


__all__ = [
    "GraphAuxiliaryLossConfig",
    "graph_auxiliary_loss",
    "require_active_dsa_graph_objective",
    "validate_runtime_graph_contract",
]
