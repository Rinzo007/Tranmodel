"""Run one AequilibraE transit assignment per operating period."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .interval_profile import DEFAULT_INTERVAL_PROFILE
from .model import Evaluation, Route, RouteSet
from .period_assignment import PeriodAssignment
from .aequilibrae_eval import evaluate_route_set_aequilibrae


def _period_route_set(route_set: RouteSet, frequency_factor: float) -> RouteSet:
    """Apply the period frequency multiplier while preserving route metadata."""
    return RouteSet([
        Route(r.nodes, r.route_id, max(0.1, r.frequency_vph * frequency_factor), r.max_section_flow_pph, r.vehicle_type)
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
    """Evaluate a route set across all six periods.

    Each period gets its own OD matrix and GTFS frequency scaling, so the
    resulting transit assignment is genuinely period-specific. The returned
    passenger-service metrics are weighted by period hours and demand factor.
    Operating-cost fields remain the peak/base-network costs; detailed
    per-period costs are retained in metadata and are aggregated separately
    by the operating-plan layer.
    """
    notify = progress or (lambda _msg: None)
    period_rows: list[PeriodAssignment] = []
    period_meta: list[dict] = []
    total_weight = 0.0
    weighted = {k: 0.0 for k in ("user_cost", "waiting_time", "walking_time", "transfers", "direct_demand_share")}
    total_uncovered = 0.0
    max_capacity_excess = 0.0

    for p in DEFAULT_INTERVAL_PROFILE:
        demand_factor = float((demand_factors or {}).get(p.name, p.frequency_factor))
        period_demand = np.asarray(base_demand, dtype=float) * max(demand_factor, 0.0)
        period_routes = _period_route_set(route_set, p.frequency_factor)
        notify(f"AequilibraE: период {p.name} {p.start}–{p.end}, коэффициент спроса {demand_factor:.2f}, частоты ×{p.frequency_factor:.2f}")
        # Separate cache namespace prevents a period assignment from being
        # confused with another period that uses the same route geometry.
        period_cache = Path(cache_dir) / "periods" / p.name.replace(":", "-").replace(" ", "_") if cache_dir else None
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
        )
        weight = max(p.hours * max(demand_factor, 0.0), 0.0)
        total_weight += weight
        weighted["user_cost"] += ev.user_cost * weight
        weighted["transfers"] += ev.transfers * weight
        weighted["direct_demand_share"] += ev.direct_demand_share * weight
        meta = dict(ev.metadata or {})
        weighted["waiting_time"] += float(meta.get("waiting_time", 0.0) or 0.0) * weight
        weighted["walking_time"] += float(meta.get("walking_time", 0.0) or 0.0) * weight
        total_uncovered += ev.uncovered_demand
        max_capacity_excess = max(max_capacity_excess, ev.capacity_excess)
        period_meta.append({"name": p.name, "start": p.start, "end": p.end, "hours": p.hours, "frequency_factor": p.frequency_factor, "demand_factor": demand_factor, "evaluation": asdict(ev)})

    norm = total_weight or 1.0
    meta = {
        "evaluator": "AequilibraE-6-period",
        "period_count": 6,
        "periods": period_meta,
        "periods_weighted_by": "hours × demand_factor",
        "period_frequency_profile": [{"name": p.name, "start": p.start, "end": p.end, "factor": p.frequency_factor, "hours": p.hours} for p in DEFAULT_INTERVAL_PROFILE],
    }
    # Preserve the economic/operating plan from the base route set using the
    # peak assignment envelope. Period-level details remain available above.
    best_base = min((x["evaluation"] for x in period_meta), key=lambda x: float(x.get("user_cost", 1e30)), default={})
    best_meta = dict(best_base.get("metadata") or {})
    meta.update(best_meta)
    meta["period_results"] = period_meta
    return Evaluation(
        score=0.0,
        user_cost=weighted["user_cost"] / norm,
        operator_cost=float(best_base.get("operator_cost", 0.0)),
        uncovered_demand=total_uncovered,
        transfers=weighted["transfers"] / norm,
        direct_demand_share=weighted["direct_demand_share"] / norm,
        capacity_excess=max_capacity_excess,
        metadata=meta,
    )
