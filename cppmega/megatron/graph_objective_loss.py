"""Differentiable graph supervision for production DSA indexer scores."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import torch
import torch.nn.functional as F

from cppmega.megatron.graph_recipe import (
    STAGE1_GRAPH_RELATIONS,
    stage1_graph_config_kwargs,
    validate_stage1_graph_total_loss_contract,
)


GRAPH_BIAS_BETA_ENV = "CPPMEGA_GRAPH_BIAS_BETA"
GRAPH_BIAS_BETA_LEGACY_ENVS = (
    "CPPMEGA_DSA_GRAPH_BIAS_BETA",
    "CPPMEGA_GRAPH_ATTENTION_BIAS_BETA",
)


def validate_graph_bias_beta(
    value: object, *, source: str = "graph bias beta"
) -> float:
    try:
        beta = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} must be a finite positive scalar, got {value!r}"
        ) from exc
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"{source} must be a finite positive scalar, got {value!r}")
    return beta


def resolve_graph_bias_beta(
    environment: Mapping[str, str] | None = None,
) -> float:
    """Resolve one beta for DSA selection, dense attention, and graph loss.

    ``CPPMEGA_GRAPH_BIAS_BETA`` is the canonical name. The historical DSA and
    dense-attention names remain accepted as aliases, but every present alias
    must represent the exact same rational value.
    """

    source = os.environ if environment is None else environment
    values: list[tuple[str, float]] = []
    for name in (GRAPH_BIAS_BETA_ENV, *GRAPH_BIAS_BETA_LEGACY_ENVS):
        raw = source.get(name)
        if raw is None or not raw.strip():
            continue
        values.append(
            (
                name,
                validate_graph_bias_beta(
                    raw,
                    source=f"graph bias beta ({name})",
                ),
            )
        )

    if not values:
        return validate_graph_bias_beta(
            stage1_graph_config_kwargs()["bias_beta"],
            source="Stage-1 graph recipe bias beta",
        )

    _, first_value = values[0]
    for _, value in values[1:]:
        if Fraction(str(value)) != Fraction(str(first_value)):
            details = ", ".join(
                f"{value_name}={value_value:g}"
                for value_name, value_value in values
            )
            raise ValueError(
                "graph bias beta knobs differ; all aliases must match exactly: "
                f"{details}"
            )
    return first_value


def graph_bias_beta_binding(beta: float | None = None) -> dict[str, object]:
    """Return the exact beta binding carried by runtime receipts."""

    value = (
        resolve_graph_bias_beta()
        if beta is None
        else validate_graph_bias_beta(beta)
    )
    return {
        "canonical_env": GRAPH_BIAS_BETA_ENV,
        "legacy_envs": list(GRAPH_BIAS_BETA_LEGACY_ENVS),
        "value": str(Fraction(str(value))),
    }


def compose_dsa_indexer_total_loss(
    indexer_loss: torch.Tensor,
    graph_loss: torch.Tensor,
) -> torch.Tensor:
    """Compose the one DSA loss scalar carried by Megatron's total backward.

    The returned scalar is attached to the attention output by
    ``DSAIndexerLossAutoScaler``. Requiring both components here prevents the
    graph objective from becoming a detached logging-only or post-hoc path.
    """

    components = {
        "indexer_loss": indexer_loss,
        "graph_loss": graph_loss,
    }
    for name, value in components.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor, got {tuple(value.shape)}")
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point, got {value.dtype}")
        if not bool(torch.isfinite(value.detach()).item()):
            raise ValueError(f"{name} must be finite")
    if indexer_loss.device != graph_loss.device:
        raise ValueError(
            "indexer_loss and graph_loss must be on the same device: "
            f"{indexer_loss.device} != {graph_loss.device}"
        )
    if indexer_loss.dtype != graph_loss.dtype:
        raise ValueError(
            "indexer_loss and graph_loss must use the same dtype: "
            f"{indexer_loss.dtype} != {graph_loss.dtype}"
        )
    return indexer_loss + graph_loss


@dataclass(frozen=True)
class GraphAuxiliaryLossConfig:
    global_weight: float
    bce_weight: float
    coverage_weight: float
    topk: int
    indexer_weight: float = 0.001
    layer_weight: float = 1.0
    pos_weight: float = 1.0
    margin: float = 1.0
    bias_beta: float = 1.0
    relations: tuple[str, ...] = STAGE1_GRAPH_RELATIONS

    def __post_init__(self) -> None:
        for name in (
            "global_weight",
            "indexer_weight",
            "layer_weight",
            "bce_weight",
            "coverage_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.topk < 1:
            raise ValueError("topk must be >= 1")
        if not math.isfinite(float(self.pos_weight)) or self.pos_weight <= 0.0:
            raise ValueError("pos_weight must be > 0")
        if not math.isfinite(float(self.margin)) or self.margin < 0.0:
            raise ValueError("margin must be >= 0")
        if not math.isfinite(float(self.bias_beta)) or self.bias_beta <= 0.0:
            raise ValueError("bias beta (bias_beta) must be > 0")
        if not self.relations or any(not relation for relation in self.relations):
            raise ValueError("relations must contain at least one non-empty name")
        if len(set(self.relations)) != len(self.relations):
            raise ValueError("relations must not contain duplicates")

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "GraphAuxiliaryLossConfig":
        source = os.environ if environment is None else environment
        defaults = stage1_graph_config_kwargs()
        relations = tuple(
            item.strip()
            for item in source.get(
                "CPPMEGA_DSA_GRAPH_AUX_RELATIONS",
                ",".join(STAGE1_GRAPH_RELATIONS),
            ).split(",")
            if item.strip()
        )
        try:
            return cls(
                global_weight=float(
                    source.get(
                        "CPPMEGA_DSA_GRAPH_AUX_WEIGHT",
                        str(defaults["global_weight"]),
                    )
                ),
                indexer_weight=float(
                    source.get(
                        "CPPMEGA_DSA_INDEXER_LOSS_COEFF",
                        str(defaults["indexer_weight"]),
                    )
                ),
                layer_weight=float(
                    source.get(
                        "CPPMEGA_DSA_GRAPH_LAYER_WEIGHT",
                        str(defaults["layer_weight"]),
                    )
                ),
                bce_weight=float(
                    source.get(
                        "CPPMEGA_DSA_GRAPH_BCE_WEIGHT",
                        str(defaults["bce_weight"]),
                    )
                ),
                coverage_weight=float(
                    source.get(
                        "CPPMEGA_DSA_GRAPH_COVERAGE_WEIGHT",
                        str(defaults["coverage_weight"]),
                    )
                ),
                topk=int(
                    source.get(
                        "CPPMEGA_DSA_GRAPH_AUX_TOPK", str(defaults["topk"])
                    )
                ),
                pos_weight=float(
                    source.get(
                        "CPPMEGA_DSA_GRAPH_POS_WEIGHT",
                        str(defaults["pos_weight"]),
                    )
                ),
                margin=float(
                    source.get(
                        "CPPMEGA_DSA_GRAPH_MARGIN", str(defaults["margin"])
                    )
                ),
                bias_beta=float(
                    resolve_graph_bias_beta(source)
                ),
                relations=relations,
            )
        except ValueError as exc:
            raise ValueError(f"invalid DSA graph auxiliary environment: {exc}") from exc


def validate_runtime_graph_contract(
    graph_contract: Mapping[str, object],
    *,
    environment: Mapping[str, str] | None = None,
    require_included_auxiliary: bool = True,
) -> None:
    """Require runtime graph-loss knobs to exactly match the data receipt."""

    if not _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED", environment=environment):
        raise ValueError(
            "production graph objective requires CPPMEGA_GRAPH_ROUTES_ENABLED=1"
        )
    if not _env_flag("CPPMEGA_STRUCTURE_ENABLED", environment=environment):
        raise ValueError(
            "production graph objective requires CPPMEGA_STRUCTURE_ENABLED=1"
        )
    validate_stage1_graph_total_loss_contract(graph_contract)
    requested = graph_objective_requested(environment=environment)
    if (
        require_included_auxiliary
        and graph_contract.get("included_in_total_loss") is True
        and not requested
    ):
        raise ValueError(
            "graph_auxiliary.included_in_total_loss=true requires the DSA graph "
            "auxiliary objective to be enabled"
        )
    # Route-only runtime validation may train dense attention through LM loss.
    # Production preflight passes require_included_auxiliary=True and rejects
    # that mode for contracts which promise inclusion in total loss.
    if not requested:
        return
    config = GraphAuxiliaryLossConfig.from_env(environment)
    expected_relations = tuple(graph_contract.get("relations", ()))
    if config.relations != expected_relations:
        raise ValueError(
            "graph auxiliary runtime relations differ from contract: "
            f"runtime={config.relations}, contract={expected_relations}"
        )
    comparisons = {
        "global_weight": config.global_weight,
        "indexer_weight": config.indexer_weight,
        "layer_weight": config.layer_weight,
        "bce_weight": config.bce_weight,
        "coverage_weight": config.coverage_weight,
        "pos_weight": config.pos_weight,
        "margin": config.margin,
        "bias_beta": config.bias_beta,
    }
    for field, runtime_value in comparisons.items():
        raw_contract = graph_contract.get(field)
        if not isinstance(raw_contract, str):
            raise ValueError(f"graph_auxiliary.{field} contract value must be exact")
        if Fraction(str(runtime_value)) != Fraction(raw_contract):
            display_name = "bias beta" if field == "bias_beta" else field
            raise ValueError(
                f"graph auxiliary runtime {display_name}={runtime_value} differs from "
                f"contract={raw_contract}"
            )
    if config.topk != graph_contract.get("topk"):
        raise ValueError(
            f"graph auxiliary runtime topk={config.topk} differs from "
            f"contract={graph_contract.get('topk')!r}"
        )
    if graph_contract.get("layer_reduction") != "sum":
        raise ValueError("graph auxiliary layer_reduction must be 'sum'")


def _env_flag(
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environment is None else environment
    raw = source.get(name, "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} has invalid boolean value {raw!r}")


def graph_objective_requested(
    *,
    environment: Mapping[str, str] | None = None,
) -> bool:
    return _env_flag(
        "CPPMEGA_DSA_GRAPH_AUX_ENABLED",
        environment=environment,
    )


def require_active_dsa_graph_objective(
    transformer_config: object,
    *,
    required: bool = True,
) -> None:
    """Fail before model construction if graph loss cannot reach total loss."""

    if not required:
        return
    if not graph_objective_requested():
        raise ValueError(
            "DSA graph objective requested but CPPMEGA_DSA_GRAPH_AUX_ENABLED is disabled"
        )
    if not _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED"):
        raise ValueError(
            "DSA graph objective requested but CPPMEGA_GRAPH_ROUTES_ENABLED is disabled"
        )
    if not _env_flag("CPPMEGA_STRUCTURE_ENABLED"):
        raise ValueError(
            "graph objective requested but CPPMEGA_STRUCTURE_ENABLED is disabled"
        )
    graph_config = GraphAuxiliaryLossConfig.from_env()
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
    if Fraction(str(coefficient_value)) != Fraction(str(graph_config.indexer_weight)):
        raise ValueError(
            "TransformerConfig.dsa_indexer_loss_coeff differs from the graph "
            f"indexer coefficient: {coefficient_value} != "
            f"{graph_config.indexer_weight}"
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
    positive_non_finite = (
        (edge_targets > 0)
        & (pair_mask > 0)
        & ~torch.isfinite(indexer_scores)
    )
    if torch.any(positive_non_finite):
        raise ValueError(
            "positive graph target has a non-finite indexer score; "
            "refusing invalid graph batch"
        )
    return tuple(int(value) for value in indexer_scores.shape)


def _finite_connected_zero(scores: torch.Tensor) -> torch.Tensor:
    finite_scores = torch.where(
        torch.isfinite(scores),
        scores,
        torch.zeros_like(scores),
    )
    return finite_scores.sum().mul(0.0)


def graph_auxiliary_loss(
    indexer_scores: torch.Tensor,
    edge_targets: torch.Tensor,
    *,
    pair_mask: torch.Tensor,
    config: GraphAuxiliaryLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return weighted graph loss and exact differentiable components.

    Samples without positive graph edges are excluded rather than assigned
    fabricated negatives. A positive graph target with a non-finite score is an
    invalid batch and raises before any finite-score filtering. A fully
    graph-ineligible batch returns a connected zero scalar; dataset-level
    eligibility is enforced by the objective receipt.
    """

    _batch, _queries, keys = _validate_inputs(indexer_scores, edge_targets, pair_mask)
    scores = indexer_scores.float()
    targets = edge_targets.to(device=scores.device, dtype=torch.float32)
    finite_scores = torch.isfinite(scores)
    safe_scores = torch.where(finite_scores, scores, torch.zeros_like(scores))
    valid_pairs = pair_mask.to(device=scores.device, dtype=torch.bool) & finite_scores
    positive_edges = targets[valid_pairs].sum()
    if int(positive_edges.detach().item()) == 0:
        zero = _finite_connected_zero(scores)
        return zero, {
            "bce": zero,
            "coverage": zero,
            "positive_edges": positive_edges,
        }

    eligible_samples = ((targets > 0) & valid_pairs).any(dim=(1, 2))
    valid_pairs = valid_pairs & eligible_samples[:, None, None]
    element_bce = F.binary_cross_entropy_with_logits(
        safe_scores,
        targets,
        reduction="none",
        pos_weight=torch.tensor(
            config.pos_weight, device=scores.device, dtype=scores.dtype
        ),
    )
    bce = element_bce[valid_pairs].mean()

    if config.topk >= keys:
        coverage = _finite_connected_zero(scores)
    else:
        masked_scores = safe_scores.masked_fill(~valid_pairs, float("-inf"))
        boundary = torch.topk(masked_scores, k=config.topk + 1, dim=-1).values[..., -1:]
        finite_boundary = torch.isfinite(boundary)
        deficits = torch.relu(boundary + config.margin - safe_scores)
        penalties = (
            deficits
            * targets
            * valid_pairs.to(scores.dtype)
            * finite_boundary.to(scores.dtype)
        )
        coverage = penalties.sum() / positive_edges.clamp_min(1.0)

    unscaled = config.global_weight * config.layer_weight * (
        config.bce_weight * bce + config.coverage_weight * coverage
    )
    total = config.indexer_weight * unscaled
    return total, {
        "bce": bce,
        "coverage": coverage,
        "positive_edges": positive_edges,
    }


__all__ = [
    "GraphAuxiliaryLossConfig",
    "GRAPH_BIAS_BETA_ENV",
    "GRAPH_BIAS_BETA_LEGACY_ENVS",
    "compose_dsa_indexer_total_loss",
    "graph_bias_beta_binding",
    "graph_objective_requested",
    "graph_auxiliary_loss",
    "require_active_dsa_graph_objective",
    "resolve_graph_bias_beta",
    "validate_graph_bias_beta",
    "validate_runtime_graph_contract",
]
