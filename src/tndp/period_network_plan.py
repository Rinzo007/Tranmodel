"""Reconcile six period assignments into one coherent annual network plan."""
from __future__ import annotations
from typing import Iterable
from .cost_aggregation import aggregate_network_costs
from .period_vehicle_plan import build_route_vehicle_plan_auto


def _period_flow_rows(period_results: Iterable[dict]) -> dict[str, list[float]]:
    flows: dict[str, list[float]] = {}
    for period in period_results:
        evaluation = period.get("evaluation", {}) if isinstance(period, dict) else {}
        metadata = evaluation.get("metadata", {}) if isinstance(evaluation, dict) else {}
        rows = metadata.get("route_characteristics", []) or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("route_id", ""))
            if not rid:
                continue
            flows.setdefault(rid, []).append(float(row.get("max_section_flow_pph", 0.0) or 0.0))
    return flows


def reconcile_period_network(route_specs: Iterable[dict], period_results: Iterable[dict], *, periods,
                              annual_days: int = 350, park_trip_coefficient: float = 0.90,
                              speed_kmh: float = 18.0, interval_reserve_sec: float = 20.0,
                              terminal_delay_reserve: float = 0.08,
                              charging_min_per_terminal: float = 10.0,
                              allowed_vehicle_types=None) -> dict:
    """Build one network plan, auto-selecting the least-cost vehicle per route from period Qmax."""
    period_results = list(period_results)
    route_specs = list(route_specs)
    periods = tuple(periods)
    flows = _period_flow_rows(period_results)
    default_allowed = tuple(allowed_vehicle_types or ())
    plans = []
    for spec in route_specs:
        rid = str(spec["route_id"])
        route_flows = list(flows.get(rid, []))
        fallback = float(spec.get("peak_flow_pph", 0.0) or 0.0)
        if len(route_flows) < len(periods):
            route_flows.extend([fallback] * (len(periods) - len(route_flows)))
        vehicle_types = tuple(spec.get("allowed_vehicle_types", default_allowed))
        if not vehicle_types:
            raise ValueError(f"No allowed vehicle types for route {rid}")
        plan = build_route_vehicle_plan_auto(
            route_id=rid,
            route_length_km=float(spec["route_length_km"]),
            period_peak_flows=route_flows[:len(periods)],
            allowed_vehicle_types=vehicle_types,
            periods=periods,
            speed_kmh=speed_kmh,
            interval_reserve_sec=interval_reserve_sec,
            terminal_delay_reserve=terminal_delay_reserve,
            charging_min_per_terminal=charging_min_per_terminal,
            annual_days=annual_days,
            park_trip_coefficient=park_trip_coefficient,
        )
        plans.append(plan)
    cost_rows = []
    for plan in plans:
        row = dict(plan["costs"])
        row.update({"fleet": plan["peak_fleet"], "annual_mileage_km": plan["annual_mileage_km"], "annual_hours": plan["annual_hours"]})
        cost_rows.append(row)
    network = aggregate_network_costs(cost_rows)
    return {"routes": plans, "network": network, "period_count": len(periods), "vehicle_selection": "auto_by_peak_assigned_flow"}
