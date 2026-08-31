"""Helpers for aggregating route operating plans."""
from __future__ import annotations
from .interval_profile import DEFAULT_INTERVAL_PROFILE
from .multi_period import build_period_plan, summarize_period_plan


def build_network_operating_plan(route_specs, periods=DEFAULT_INTERVAL_PROFILE, **kwargs):
    """Build and aggregate multi-period plans for all routes.

    Each route spec must contain route_length_km and vehicle_type and may contain
    peak_flow_pph and route_id. Fleet and lifecycle costs are based on the
    simultaneous peak fleet of each route, not the sum of fleets in periods.
    """
    routes = []
    for i, spec in enumerate(route_specs):
        vehicle_type = str(spec["vehicle_type"])
        plan = build_period_plan(
            route_length_km=float(spec["route_length_km"]),
            peak_flow_pph=float(spec.get("peak_flow_pph", 0.0)),
            vehicle_type=vehicle_type,
            periods=periods,
            **kwargs,
        )
        summary = summarize_period_plan(plan, vehicle_type=vehicle_type)
        summary["route_id"] = str(spec.get("route_id", i + 1))
        routes.append(summary)
    return {
        "routes": routes,
        "peak_fleet": sum(int(r["peak_fleet"]) for r in routes),
        "daily_trips": sum(float(r["daily_trips"]) for r in routes),
        "annual_mileage_km": sum(float(r["annual_mileage_km"]) for r in routes),
        "annual_hours": sum(float(r["annual_hours"]) for r in routes),
        "annual_contract_cost_mln": sum(float(r.get("annual_contract_cost_mln", 0.0)) for r in routes),
        "annual_amortization_mln": sum(float(r.get("annual_amortization_mln", 0.0)) for r in routes),
        "one_off_fleet_cost_mln": sum(float(r.get("one_off_fleet_cost_mln", 0.0)) for r in routes),
    }
