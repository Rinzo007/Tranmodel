"""Run one AequilibraE transit assignment per operating period."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod
from .model import Evaluation, Route, RouteSet
from .period_assignment import PeriodAssignment
from .aequilibrae_eval import evaluate_route_set_aequilibrae
from .peak_fleet import reconcile_route_periods


def _period_key(p: IntervalPeriod) -> str:
    return f"{p.number}_{p.start.replace(':', '')}_{p.end.replace(':', '')}"


def _period_route_set(route_set: RouteSet, frequency_factor: float) -> RouteSet:
    """Apply period frequency multiplier while preserving route metadata."""
    return RouteSet([
        Route(r.nodes, r.route_id,
              max(0.1, r.frequency_vph * frequency_factor),
              r.max_section_flow_pph, r.vehicle_type)
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
    """Evaluate one route set with six genuinely separate transit assignments.

    Each period gets its own OD demand and frequency multiplier. After all six
    assignments, the maximum assigned section flow per route is used to build
    one common fleet envelope and a single daily/annual operating plan.
    """
    notify = progress or (lambda _msg: None)
    period_rows: list[PeriodAssignment] = []
    period_meta: list[dict] = []
    total_weight = 0.0
    weighted = {k: 0.0 for k in ("user_cost", "waiting_time", "walking_time", "transfers", "direct_demand_share")}
    total_uncovered = 0.0
    max_capacity_excess = 0.0
    route_period_flows = [[0.0 for _ in DEFAULT_INTERVAL_PROFILE] for _ in route_set.routes]

    for pi, p in enumerate(DEFAULT_INTERVAL_PROFILE):
        key = _period_key(p)
        raw_factor = None
        if demand_factors:
            raw_factor = demand_factors.get(key)
            if raw_factor is None:
                raw_factor = demand_factors.get(str(p.number))
            if raw_factor is None:
                raw_factor = demand_factors.get(p.name)
        demand_factor = float(p.frequency_factor if raw_factor is None else raw_factor)
        if demand_factor < 0:
            raise ValueError(f"Negative demand factor for period {key}")
        period_demand = np.asarray(base_demand, dtype=float) * demand_factor
        period_routes = _period_route_set(route_set, p.frequency_factor)
        notify(f"AequilibraE: период {p.name} {p.start}–{p.end}, спрос ×{demand_factor:.2f}, частота ×{p.frequency_factor:.2f}")
        period_cache = Path(cache_dir) / "periods" / key if cache_dir else None
        ev = evaluate_route_set_aequilibrae(
            period_routes, period_demand, stop_xy_lonlat, project_path, config,
            road_graph=road_graph, stop_mapping=stop_mapping, path_index=path_index,
            stop_to_zone=stop_to_zone, cache_dir=period_cache, assignment_iteration=0,
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
        period_rows.append(PeriodAssignment(
            period=key, user_cost=ev.user_cost,
            waiting_time=float(em.get("waiting_time", 0.0) or 0.0),
            walking_time=float(em.get("walking_time", 0.0) or 0.0),
            transfers=ev.transfers, uncovered_demand=ev.uncovered_demand,
            capacity_excess=ev.capacity_excess, annualized_weight=weight,
        ))
        period_meta.append({
            "period_id": key, "name": p.name, "start": p.start, "end": p.end,
            "hours": p.hours, "frequency_factor": p.frequency_factor,
            "demand_factor": demand_factor, "evaluation": asdict(ev),
        })

    norm = total_weight or 1.0
    best_base = min(period_rows, key=lambda x: x.user_cost, default=None)
    meta = {
        "evaluator": "AequilibraE-6-period",
        "period_count": len(DEFAULT_INTERVAL_PROFILE),
        "period_results": period_meta,
        "periods_weighted_by": "hours × demand_factor",
        "period_assignment_summary": {
            "user_cost": weighted["user_cost"] / norm,
            "waiting_time": weighted["waiting_time"] / norm,
            "walking_time": weighted["walking_time"] / norm,
            "transfers": weighted["transfers"] / norm,
            "uncovered_demand_hours": total_uncovered,
        },
        "period_frequency_profile": [
            {"period_id": _period_key(p), "name": p.name, "start": p.start, "end": p.end,
             "factor": p.frequency_factor, "hours": p.hours}
            for p in DEFAULT_INTERVAL_PROFILE
        ],
    }
    if best_base is not None:
        for row in period_meta:
            if row["period_id"] == best_base.period:
                meta.update(dict(row["evaluation"].get("metadata") or {}))
                break

    # Reconcile the six assigned maximum-section flows into one physical fleet
    # envelope. A route may require more vehicles in one peak than in another,
    # but the network owns only the simultaneous peak fleet.
    route_plans = []
    route_lengths = []
    route_period_details = []
    try:
        cached_characteristics = []
        for row in period_meta:
            cached_characteristics.append(row["evaluation"].get("metadata", {}).get("route_characteristics", []))
        for ri, route in enumerate(route_set.routes):
            one_way_lengths = [float(x["one_way_length_km"]) for x in cached_characteristics[0][ri:ri+1] if isinstance(x, dict) and "one_way_length_km" in x]
            length = one_way_lengths[0] if one_way_lengths else 0.0
            if length > 0:
                reconciliation = reconcile_route_periods(
                    route_length_km=length,
                    period_peak_flows=route_period_flows[ri],
                    vehicle_type=route.vehicle_type,
                    periods=DEFAULT_INTERVAL_PROFILE,
                    speed_kmh=config.speed_kmh,
                    interval_reserve_sec=config.interval_reserve_sec,
                    terminal_delay_reserve=config.terminal_delay_reserve,
                    charging_min_per_terminal=config.charging_min_per_terminal,
                    annual_days=config.annual_days,
                    park_trip_coefficient=config.park_trip_coefficient,
                )
                route_plans.append(reconciliation)
                route_period_details.append(reconciliation.get("periods", []))
                route_lengths.append(length)
    except (KeyError, IndexError, TypeError, ValueError):
        route_plans = []

    if route_plans:
        meta["route_peak_reconciliation"] = route_plans
        meta["peak_fleet_reconciled"] = int(sum(int(x.get("peak_fleet", 0)) for x in route_plans))
        meta["reconciled_annual_mileage_km"] = float(sum(float(x.get("annual_mileage_km", 0.0)) for x in route_plans))
        meta["reconciled_annual_hours"] = float(sum(float(x.get("annual_hours", 0.0)) for x in route_plans))
    return Evaluation(
        score=0.0,
        user_cost=weighted["user_cost"] / norm,
        operator_cost=float(meta.get("reconciled_annual_mileage_km", meta.get("annual_mileage_km", 0.0)) or 0.0),
        uncovered_demand=total_uncovered,
        transfers=weighted["transfers"] / norm,
        direct_demand_share=weighted["direct_demand_share"] / norm,
        capacity_excess=max_capacity_excess,
        metadata=meta,
    )
