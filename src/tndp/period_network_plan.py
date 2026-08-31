"""Reconcile six period assignments into one route and network operating plan."""
from __future__ import annotations
from typing import Iterable

from .cost_aggregation import aggregate_network_costs
from .period_vehicle_plan import build_route_vehicle_plan


def _period_flow_rows(period_results: Iterable[dict]) -> dict[str, list[float]]:
    flows: dict[str, list[float]] = {}
    for period in period_results:
        evaluation = period.get("evaluation", {}) if isinstance(period, dict) else {}
        metadata = evaluation.get("metadata", {}) if isinstance(evaluation, dict) else {}
        rows = metadata.get("route_characteristics", []) or []
        for row in rows:
            rid = str(row.get("route_id", ""))
            if not rid:
                continue
            flow = float(row.get("max_section_flow_pph", 0.0) or 0.0)
            flows.setdefault(rid, []).append(flow)
    return flows


def reconcile_period_network(
    route_specs: Iterable[dict],
    period_results: Iterable[dict],
    *,
    periods,
    annual_days: int = 350,
    park_trip_coefficient: float = 0.90,
    speed_kmh: float = 18.0,
    interval_reserve_sec: float = 20.0,
    terminal_delay_reserve: float = 0.08,
    charging_min_per_terminal: float = 10.0,
) -> dict:
    """Build one coherent annual operating plan from period-assignment flows."""
    period_results = list(period_results)
    route_specs = list(route_specs)
    flows = _period_flow_rows(period_results)
    plans = []
    for spec in route_specs:
        rid = str(spec["route_id"])
        route_flows = flows.get(rid, [float(spec.get("peak_flow_pph", 0.0))] * len(tuple(periods)))
        if len(route_flows) < len(tuple(periods)):
            route_flows = route_flows + [route_flows[-1] if route_flows else 0.0] * (len(tuple(periods)) - len(route_flows))
        plan = build_route_vehicle_plan(
            route_id=rid,
            route_length_km=float(spec["route_length_km"]),
            period_peak_flows=route_flows[:len(tuple(periods))],
            vehicle_type=str(spec["vehicle_type"]),
            periods=periods,
            speed_kmh=speed_kmh,
            interval_reserve_sec=interval_reserve_sec,
            terminal_delay_reserve=terminal_delay_reserve,
            charging_min_per_terminal=charging_min_per_terminal,
            annual_days=annual_days,
            park_trip_coefficient=park_trip_coefficient,
        )
        plans.append(plan)
    cost_routes = [p["costs"] | {"fleet": p["peak_fleet"], "annual_mileage_km": p["annual_mileage_km"], "annual_hours": p["annual_hours"]} for p in plans]
    network_costs = aggregate_network_costs(cost_routes)
    return {"routes": plans, "network": network_costs, "period_count": len(tuple(periods))}
