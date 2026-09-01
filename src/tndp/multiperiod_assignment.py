"""Run one AequilibraE transit assignment per operating period."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod
from .model import Evaluation, Route, RouteSet
from .period_assignment import PeriodAssignment
from .aequilibrae_eval import evaluate_route_set_aequilibrae
from .period_vehicle_plan import build_network_vehicle_plan


def _period_key(p: IntervalPeriod) -> str:
    return f"{p.number}_{p.start.replace(':', '')}_{p.end.replace(':', '')}"


def _period_route_set(route_set: RouteSet, frequency_factor: float) -> RouteSet:
    """Return the service plan for a period.

    ``frequency_factor`` belongs exclusively to the service-frequency profile.
    It must never be reused as a demand multiplier.
    """
    return RouteSet([
        Route(
            r.nodes,
            r.route_id,
            max(0.1, r.frequency_vph * frequency_factor),
            r.max_section_flow_pph,
            r.vehicle_type,
        )
        for r in route_set.routes
    ])


def evaluate_route_set_aequilibrae_periods(
    route_set: RouteSet,
    base_demand: np.ndarray,
    stop_xy_lonlat: np.ndarray,
    project_path: str | Path,
    config,
    *,
    road_graph,
    stop_mapping,
    path_index=None,
    stop_to_zone=None,
    cache_dir=None,
    demand_factors: dict[str, float] | None = None,
    progress=None,
) -> Evaluation:
    """Evaluate a route set in six periods and build one unified annual operating plan.

    The interval profile controls service frequency only. A separate
    ``demand_factors`` mapping may optionally model a temporal demand profile;
    when it is absent, demand is identical in all periods.
    """
    notify = progress or (lambda _msg: None)
    period_meta, period_rows = [], []
    total_weight = 0.0
    weighted = {k: 0.0 for k in ("user_cost", "waiting_time", "walking_time", "transfers", "direct_demand_share")}
    total_uncovered = 0.0
    max_capacity_excess = 0.0
    route_period_flows = [[0.0 for _ in DEFAULT_INTERVAL_PROFILE] for _ in route_set.routes]

    for pi, p in enumerate(DEFAULT_INTERVAL_PROFILE):
        key = _period_key(p)
        raw_factor = None
        if demand_factors:
            raw_factor = demand_factors.get(key, demand_factors.get(str(p.number), demand_factors.get(p.name)))
        demand_factor = float(1.0 if raw_factor is None else raw_factor)
        if demand_factor < 0:
            raise ValueError(f"Negative demand factor for period {key}")
        period_demand = np.asarray(base_demand, dtype=float) * demand_factor
        period_routes = _period_route_set(route_set, p.frequency_factor)
        notify(
            f"AequilibraE: {key} {p.name} {p.start}–{p.end}: "
            f"спрос ×{demand_factor:.2f}, частота ×{p.frequency_factor:.2f}"
        )
        period_cache = Path(cache_dir) / "periods" / key if cache_dir else None
        # The period frequency factor has already been applied to ``period_routes``.
        # Pass a neutral one-period GTFS profile so the factor is not applied twice,
        # and so AequilibraE receives only the timetable for the period being evaluated.
        gtfs_period = (
            IntervalPeriod(p.number, p.name, p.start, p.end, 1.0, p.hours),
        )
        ev = evaluate_route_set_aequilibrae(
            period_routes,
            period_demand,
            stop_xy_lonlat,
            project_path,
            config,
            road_graph=road_graph,
            stop_mapping=stop_mapping,
            path_index=path_index,
            stop_to_zone=stop_to_zone,
            cache_dir=period_cache,
            assignment_iteration=0,
            service_profile=gtfs_period,
        )
        weight = p.hours * demand_factor
        total_weight += weight
        weighted["user_cost"] += ev.user_cost * weight
        weighted["transfers"] += ev.transfers * weight
        weighted["direct_demand_share"] += ev.direct_demand_share * weight
        em = dict(ev.metadata or {})
        weighted["waiting_time"] += float(em.get("waiting_time", 0.0) or 0.0) * weight
        weighted["walking_time"] += float(em.get("walking_time", 0.0) or 0.0) * weight
        total_uncovered += ev.uncovered_demand * p.hours
        max_capacity_excess = max(max_capacity_excess, ev.capacity_excess)
        rows = em.get("route_characteristics") or []
        for ri in range(min(len(route_period_flows), len(rows))):
            route_period_flows[ri][pi] = float(rows[ri].get("max_section_flow_pph", 0.0) or 0.0)
        period_rows.append(
            PeriodAssignment(
                period=key,
                user_cost=ev.user_cost,
                waiting_time=float(em.get("waiting_time", 0.0) or 0.0),
                walking_time=float(em.get("walking_time", 0.0) or 0.0),
                transfers=ev.transfers,
                uncovered_demand=ev.uncovered_demand,
                capacity_excess=ev.capacity_excess,
                annualized_weight=weight,
            )
        )
        period_meta.append(
            {
                "period_id": key,
                "name": p.name,
                "start": p.start,
                "end": p.end,
                "hours": p.hours,
                "frequency_factor": p.frequency_factor,
                "demand_factor": demand_factor,
                "evaluation": asdict(ev),
            }
        )

    norm = total_weight or 1.0
    route_specs = []
    for ri, route in enumerate(route_set.routes):
        length = 0.0
        for pm in period_meta:
            chars = (pm["evaluation"].get("metadata") or {}).get("route_characteristics") or []
            if ri < len(chars) and chars[ri].get("one_way_length_km") is not None:
                length = float(chars[ri]["one_way_length_km"])
                break
        if length > 0:
            route_specs.append({
                "route_id": route.route_id or ri + 1,
                "route_length_km": length,
                "period_peak_flows": route_period_flows[ri],
                "vehicle_type": route.vehicle_type,
                "auto_vehicle": True,
                "allowed_vehicle_types": config.allowed_vehicle_types,
            })
    network_plan = (
        build_network_vehicle_plan(
            route_specs,
            speed_kmh=config.speed_kmh,
            interval_reserve_sec=config.interval_reserve_sec,
            terminal_delay_reserve=config.terminal_delay_reserve,
            charging_min_per_terminal=config.charging_min_per_terminal,
            annual_days=config.annual_days,
            park_trip_coefficient=config.park_trip_coefficient,
            frequency_profile=tuple((p.hours, p.frequency_factor) for p in DEFAULT_INTERVAL_PROFILE),
        )
        if route_specs
        else {"routes": [], "fleet": 0, "annual_mileage_km": 0.0, "annual_hours": 0.0, "costs": {}}
    )
    costs = network_plan.get("costs", {})
    meta = {
        "evaluator": "AequilibraE-6-period",
        "period_count": len(DEFAULT_INTERVAL_PROFILE),
        "period_results": period_meta,
        "periods_weighted_by": "hours × explicit demand_factor",
        "period_assignment_summary": {
            "user_cost": weighted["user_cost"] / norm,
            "waiting_time": weighted["waiting_time"] / norm,
            "walking_time": weighted["walking_time"] / norm,
            "transfers": weighted["transfers"] / norm,
            "uncovered_demand_hours": total_uncovered,
        },
        "period_frequency_profile": [
            {"period_id": _period_key(p), "name": p.name, "start": p.start, "end": p.end, "factor": p.frequency_factor, "hours": p.hours}
            for p in DEFAULT_INTERVAL_PROFILE
        ],
        "unified_operating_plan": network_plan,
        "fleet": int(network_plan.get("fleet", 0)),
        "annual_mileage_km": float(network_plan.get("annual_mileage_km", 0.0)),
        "annual_in_service_hours": float(network_plan.get("annual_hours", 0.0)),
        "annual_fuel_energy_mln": float(costs.get("fuel_energy_mln", 0.0)),
        "annual_repair_mln": float(costs.get("repair_mln", 0.0)),
        "annual_crew_cost_mln": float(costs.get("crew_mln", 0.0)),
        "annual_infrastructure_mln": float(costs.get("infrastructure_mln", 0.0)),
        "annual_dispatch_mln": float(costs.get("dispatch_mln", 0.0)),
        "annual_contract_cost_mln": float(costs.get("contract_mln", 0.0)),
        "annual_amortization_mln": float(costs.get("amortization_mln", 0.0)),
        "annual_total_cost_mln": float(costs.get("total_annual_mln", 0.0)),
        "peak_fleet_reconciled": int(network_plan.get("fleet", 0)),
        "reconciled_annual_mileage_km": float(network_plan.get("annual_mileage_km", 0.0)),
        "reconciled_annual_hours": float(network_plan.get("annual_hours", 0.0)),
    }
    return Evaluation(
        score=0.0,
        user_cost=weighted["user_cost"] / norm,
        operator_cost=float(network_plan.get("annual_mileage_km", 0.0)),
        uncovered_demand=total_uncovered,
        transfers=weighted["transfers"] / norm,
        direct_demand_share=weighted["direct_demand_share"] / norm,
        capacity_excess=max_capacity_excess,
        metadata=meta,
    )
