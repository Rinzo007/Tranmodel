"""Transparent multi-criteria objective and hard constraints for TNDP."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import isfinite

from .model import Evaluation, NetworkDesignConfig, RouteSet
from .operating_costs import annual_route_costs

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
    contract_cost: float = 0.0
    amortization: float = 0.0


def combine_objective(c: ObjectiveComponents, config: NetworkDesignConfig) -> float:
    return (c.user_cost * config.objective_user_time_weight + c.walk_cost * config.objective_walk_weight
        + c.wait_cost * config.objective_wait_weight + c.transfers * config.objective_transfer_weight
        + c.uncovered_demand * config.objective_uncovered_weight + c.capacity_excess * config.objective_overload_weight
        + c.operator_cost * config.objective_operating_weight + c.duplication * config.objective_route_duplication_weight
        + c.contract_cost * config.objective_contract_weight + c.amortization * config.objective_amortization_weight)


def evaluation_from_components(c: ObjectiveComponents, config: NetworkDesignConfig, *, metadata=None) -> Evaluation:
    return Evaluation(score=combine_objective(c, config), user_cost=c.user_cost, operator_cost=c.operator_cost,
        uncovered_demand=c.uncovered_demand, transfers=c.transfers, capacity_excess=c.capacity_excess, metadata=metadata or {})


def _derive_network_costs(route_set: RouteSet, meta: dict) -> dict[str, float]:
    """Derive missing mileage-dependent costs from route characteristics."""
    routes = meta.get("route_characteristics") or []
    totals = {"fuel_energy_mln": 0.0, "repair_mln": 0.0, "crew_mln": 0.0,
              "infrastructure_mln": 0.0, "dispatch_mln": 0.0}
    for i, route in enumerate(route_set.routes):
        rc = routes[i] if i < len(routes) and isinstance(routes[i], dict) else {}
        annual_km = float(rc.get("annual_mileage_km", 0.0) or 0.0)
        annual_hours = float(rc.get("annual_in_service_hours", 0.0) or 0.0)
        fleet = int(float(rc.get("fleet", 0) or 0))
        code = getattr(route, "vehicle_type", None)
        if not code or annual_km <= 0:
            continue
        costs = annual_route_costs(code, annual_km, fleet, annual_hours)
        for key in totals: totals[key] += costs[key]
    return totals


def apply_objective(route_set: RouteSet, evaluation: Evaluation, config: NetworkDesignConfig) -> Evaluation:
    """Apply policy weights and hard service constraints to a physical evaluation."""
    meta = dict(evaluation.metadata or {})
    derived = _derive_network_costs(route_set, meta)
    for key, value in derived.items(): meta.setdefault(key, value)
    contract = float(meta.get("annual_contract_cost_mln", 0.0) or 0.0)
    amortization = float(meta.get("annual_amortization_mln", 0.0) or 0.0)
    walk = float(meta.get("walking_cost", meta.get("walking_time", 0.0)) or 0.0)
    wait = float(meta.get("waiting_cost", meta.get("waiting_time", 0.0)) or 0.0)
    cost_total = sum(float(meta.get(k, 0.0) or 0.0) for k in ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln"))
    operator_cost = max(0.0, float(evaluation.operator_cost)) + cost_total
    duplication = float(max(0, route_set.route_count() - len(route_set.unique_undirected_signatures())))
    c = ObjectiveComponents(max(0.0, float(evaluation.user_cost)), operator_cost,
        max(0.0, float(evaluation.uncovered_demand)), max(0.0, float(evaluation.transfers)),
        max(0.0, float(evaluation.capacity_excess)), duplication, max(0.0, walk), max(0.0, wait),
        max(0.0, contract), max(0.0, amortization))
    violations: list[str] = []
    if route_set.route_count() < config.min_routes: violations.append(f"routes<{config.min_routes}")
    if route_set.route_count() > config.max_routes: violations.append(f"routes>{config.max_routes}")
    if evaluation.direct_demand_share < config.min_direct_demand_share: violations.append(f"direct_share<{config.min_direct_demand_share:.3f}")
    if evaluation.transfers > config.max_average_transfers: violations.append(f"average_transfers>{config.max_average_transfers:.3f}")
    if contract > config.max_annual_contract_cost_mln: violations.append(f"annual_contract_cost>{config.max_annual_contract_cost_mln:.3f}")
    fleet = int(float(meta.get("fleet", 0) or 0))
    if fleet > config.max_fleet: violations.append(f"fleet>{config.max_fleet}")
    coverage = float(meta.get("coverage_share", 1.0) or 0.0)
    if coverage < config.min_coverage_share: violations.append(f"coverage<{config.min_coverage_share:.3f}")
    for i, route in enumerate(route_set.routes):
        if route.frequency_vph < config.min_frequency_vph: violations.append(f"route[{i}].frequency<min")
        if route.frequency_vph > config.max_frequency_vph: violations.append(f"route[{i}].frequency>max")
        if len(route.nodes) < config.min_stops: violations.append(f"route[{i}].stops<min")
        if len(route.nodes) > config.max_stops: violations.append(f"route[{i}].stops>max")
    penalty = 1e9 + 1e6 * len(violations) if violations else 0.0
    base = combine_objective(c, config); score = base + penalty
    if not isfinite(score): score = 1e15; violations.append("non_finite_objective")
    meta.update({"objective_components": asdict(c), "objective_base_score": base, "objective_penalty": penalty,
                 "feasible": not violations, "constraint_violations": violations, "annual_operating_cost_mln": cost_total})
    return Evaluation(score=float(score), user_cost=evaluation.user_cost, operator_cost=operator_cost,
        uncovered_demand=evaluation.uncovered_demand, transfers=evaluation.transfers,
        direct_demand_share=evaluation.direct_demand_share, capacity_excess=evaluation.capacity_excess, metadata=meta)
