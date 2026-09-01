"""Canonical route economics implementation used by all TNDP cost paths."""
from __future__ import annotations
from .vehicle_types import calculate_route_operations, VEHICLE_TYPES
from .cost_aggregation import aggregate_route_costs

DEFAULT_PROFILE = ((1.0, 0.8), (2.0, 1.0), (7.5, 0.8), (3.0, 1.0), (1.5, 0.8), (3.0, 0.5))


def calculate_annual_route_economics(*, vehicle_type: str, route_length_km: float, max_section_flow_pph: float,
    speed_kmh: float = 18.0, interval_reserve_sec: float = 20.0, terminal_delay_reserve: float = 0.08,
    charging_min_per_terminal: float = 10.0, annual_days: int = 350, park_trip_coefficient: float = 0.90,
    frequency_profile=None) -> dict:
    """Calculate route operations and one non-overlapping annual cost total."""
    if vehicle_type not in VEHICLE_TYPES:
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")
    if route_length_km <= 0 or max_section_flow_pph < 0:
        raise ValueError("route_length_km must be > 0 and max_section_flow_pph must be >= 0")
    profile = tuple(frequency_profile or DEFAULT_PROFILE)
    op = calculate_route_operations(route_length_km=route_length_km, max_section_flow_pph=max_section_flow_pph,
        vehicle_type=vehicle_type, speed_kmh=speed_kmh, interval_reserve_sec=interval_reserve_sec,
        terminal_delay_reserve=terminal_delay_reserve, charging_min_per_terminal=charging_min_per_terminal,
        annual_days=annual_days, park_trip_coefficient=park_trip_coefficient, frequency_profile=profile)
    costs = aggregate_route_costs(vehicle_type=vehicle_type, annual_km=op["annual_mileage_km"],
        fleet=int(op["fleet"]), annual_hours=op["annual_in_service_hours"], route_length_km=route_length_km,
        annual_contract_mln=op.get("annual_fleet_contract_cost_mln", 0.0),
        annual_amortization_mln=op.get("annual_fleet_amortization_mln", 0.0))
    total = costs["total_annual_mln"]
    return {**op, "route_length_km": float(route_length_km), "economics": costs,
            "annual_total_cost_mln": total, "cost_per_km_rub": costs["cost_per_km_rub"],
            "cost_per_daily_trip_rub": total * 1_000_000 / max(float(op["daily_trips"]), 1e-9) / max(annual_days, 1)}
