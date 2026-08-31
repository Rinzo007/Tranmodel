"""Multi-criteria TNDP objective helpers.

All components are expressed in passenger/minute, passenger equivalents,
or million currency units so they can be combined with explicit weights.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import Evaluation, NetworkDesignConfig


@dataclass(frozen=True, slots=True)
class ObjectiveComponents:
    user_cost: float = 0.0
    operator_cost: float = 0.0
    uncovered_demand: float = 0.0
    transfers: float = 0.0
    capacity_excess: float = 0.0
    duplication: float = 0.0
    walk_cost: float = 0.0
    wait_cost: float = 0.0


def combine_objective(c: ObjectiveComponents, config: NetworkDesignConfig) -> float:
    """Return the scalar TNDP objective used by both fast and exact stages."""
    return (
        c.user_cost
        + c.walk_cost * config.walk_weight
        + c.wait_cost * config.wait_weight
        + c.transfers * config.transfer_weight * config.transfer_penalty_min
        + c.uncovered_demand * config.uncovered_demand_weight
        + c.capacity_excess * config.capacity_excess_weight
        + c.operator_cost * config.operator_route_km_weight
        + c.duplication * config.duplication_weight
    )


def evaluation_from_components(c: ObjectiveComponents, config: NetworkDesignConfig, *, metadata=None) -> Evaluation:
    score = combine_objective(c, config)
    return Evaluation(
        score=score,
        user_cost=c.user_cost,
        operator_cost=c.operator_cost,
        uncovered_demand=c.uncovered_demand,
        transfers=c.transfers,
        capacity_excess=c.capacity_excess,
        metadata=metadata or {},
    )
