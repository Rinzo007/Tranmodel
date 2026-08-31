"""Build a route vehicle/frequency plan from six period assignments.

The assignment layer supplies the maximum section flow for every route and
period. This module turns those flows into one consistent daily operating plan:
period frequency, interval, release, peak fleet, mileage, hours and annual cost.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Sequence

from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod
from .peak_fleet import reconcile_route_periods
from .cost_aggregation import aggregate_route_costs
from .vehicle_types import VEHICLE_TYPES


def build_route_vehicle_plan(
    *,
    route_id: str,
    route_length_km: float,
    period_peak_flows: Sequence[float],
    vehicle_type: str,
    periods: Sequence[IntervalPeriod] = DEFAULT_INTERVAL_PROFILE,
    speed_kmh: float = 18.0,
    interval_reserve_sec: float = 20.0,
    terminal_delay_reserve: float = 0.08,
    charging_min_per_terminal: float = 10.0,
    annual_days: int = 350,
    park_trip_coefficient: float = 0.90,
    annual_contract_mln: float = 0.0,
    annual_amortization_mln: float = 0.0,
) -> dict:
    """Create a complete daily/annual service plan for one route."""
    if vehicle_type not in VEHICLE_TYPES:
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")
    rec = reconcile_route_periods(
        route_length_km=route_length_km,
        period_peak_flows=period_peak_flows,
        vehicle_type=vehicle_type,
        periods=periods,
        speed_kmh=speed_kmh,
        interval_reserve_sec=interval_reserve_sec,
        terminal_delay_reserve=terminal_delay_reserve,
        charging_min_per_terminal=charging_min_per_terminal,
        annual_days=annual_days,
        park_trip_coefficient=park_trip_coefficient,
    )
    costs = aggregate_route_costs(
        vehicle_type=vehicle_type,
        annual_km=rec["annual_mileage_km"],
        fleet=rec["peak_fleet"],
        annual_hours=rec["annual_hours"],
        annual_contract_mln=annual_contract_mln,
        annual_amortization_mln=annual_amortization_mln,
    )
    return {
        "route_id": str(route_id),
        "route_length_km": float(route_length_km),
        "vehicle_type": vehicle_type,
        "vehicle_name": rec["vehicle_name"],
        "peak_fleet": int(rec["peak_fleet"]),
        "annual_mileage_km": float(rec["annual_mileage_km"]),
        "annual_hours": float(rec["annual_hours"]),
        "periods": rec["periods"],
        "costs": costs,
    }


def build_network_vehicle_plan(route_specs: Sequence[dict], **kwargs) -> dict:
    """Build route plans and aggregate annual network economics."""
    plans = [build_route_vehicle_plan(**spec, **kwargs) for spec in route_specs]
    fleet = sum(int(p["peak_fleet"]) for p in plans)
    annual_km = sum(float(p["annual_mileage_km"]) for p in plans)
    annual_hours = sum(float(p["annual_hours"]) for p in plans)
    cost_keys = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln",
                 "dispatch_mln", "contract_mln", "amortization_mln", "total_annual_mln")
    costs = {k: sum(float(p["costs"].get(k, 0.0)) for p in plans) for k in cost_keys}
    costs["fleet"] = fleet
    costs["annual_mileage_km"] = annual_km
    costs["annual_hours"] = annual_hours
    costs["cost_per_km_rub"] = costs["total_annual_mln"] * 1_000_000 / max(annual_km, 1e-9)
    return {"routes": plans, "fleet": fleet, "annual_mileage_km": annual_km,
            "annual_hours": annual_hours, "costs": costs}
