"""Single entry point for route operating economics.

All callers should use this adapter instead of reproducing cost formulas.
"""
from __future__ import annotations
from .cost_aggregation import aggregate_route_costs
from .vehicle_types import calculate_route_operations, VEHICLE_TYPES

REQUIRED_COST_KEYS = (
    "fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln",
    "dispatch_mln", "contract_mln", "amortization_mln", "total_annual_mln",
)

def calculate_annual_route_economics(*, vehicle_type: str, route_length_km: float,
                                     max_section_flow_pph: float, speed_kmh: float = 18.0,
                                     interval_reserve_sec: float = 20.0,
                                     terminal_delay_reserve: float = 0.08,
                                     charging_min_per_terminal: float = 10.0,
                                     annual_days: int = 350,
                                     park_trip_coefficient: float = 0.90,
                                     frequency_profile=None) -> dict:
    """Calculate operating characteristics and one authoritative annual cost."""
    if vehicle_type not in VEHICLE_TYPES:
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")
    op = calculate_route_operations(
        route_length_km=route_length_km,
        max_section_flow_pph=max_section_flow_pph,
        vehicle_type=vehicle_type,
        speed_kmh=speed_kmh,
        interval_reserve_sec=interval_reserve_sec,
        terminal_delay_reserve=terminal_delay_reserve,
        charging_min_per_terminal=charging_min_per_terminal,
        annual_days=annual_days,
        park_trip_coefficient=park_trip_coefficient,
        frequency_profile=frequency_profile or ((1.0, 0.8), (2.0, 1.0), (7.5, 0.8), (3.0, 1.0), (1.5, 0.8), (3.0, 0.5)),
    )
    costs = aggregate_route_costs(
        vehicle_type=vehicle_type,
        annual_km=op["annual_mileage_km"],
        fleet=int(op["fleet"]),
        annual_hours=op["annual_in_service_hours"],
        annual_contract_mln=op["annual_fleet_contract_cost_mln"],
        annual_amortization_mln=op["annual_fleet_amortization_mln"],
    )
    return {**op, "economics": costs}
