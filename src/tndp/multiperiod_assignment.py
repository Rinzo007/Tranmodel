"""Run one AequilibraE transit assignment per operating period."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod
from .model import Evaluation, Route, RouteSet
from .period_assignment import PeriodAssignment
from .aequilibrae_eval import evaluate_route_set_aequilibrae


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

    Demand and frequencies are period-specific. The two peak periods use the
    unscaled peak frequency, interpeak periods use 0.8, and evening uses 0.5.
    Demand factors are keyed by period number when supplied, avoiding the
    ambiguity caused by three periods sharing the name ``Межпик``.
    """
    notify = progress or (lambda _msg: None)
    period_rows: list[PeriodAssignment] = []
    period_meta: list[dict] = []
    total_weight = 0.0
    weighted = {k: 0.0 for k in ("user_cost", "waiting_time", "walking_time", "transfers", "direct_demand_share")}
    total_uncovered = 0.0
    max_capacity_excess = 0.0

    for p in DEFAULT_INTERVAL_PROFILE:
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
            {"period_id": _period_key(p), "name": p.name, "start": p.start,
             "end": p.end, "factor": p.frequency_factor, "hours": p.hours}
            for p in DEFAULT_INTERVAL_PROFILE
        ],
    }
    # Preserve route-level economic information from the best full evaluation;
    # the optimizer receives the six-period passenger metrics above.
    if best_base is not None:
        for row in period_meta:
            if row["period_id"] == best_base.period:
                meta.update(dict(row["evaluation"].get("metadata") or {}))
                break
    return Evaluation(
        score=0.0,
        user_cost=weighted["user_cost"] / norm,
        operator_cost=float(meta.get("annual_mileage_km", 0.0) or 0.0),
        uncovered_demand=total_uncovered,
        transfers=weighted["transfers"] / norm,
        direct_demand_share=weighted["direct_demand_share"] / norm,
        capacity_excess=max_capacity_excess,
        metadata=meta,
    )
