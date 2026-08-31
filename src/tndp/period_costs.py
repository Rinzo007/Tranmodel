"""Convert a six-period service plan into daily and annual operating costs."""
from __future__ import annotations
from .operating_costs import annual_route_costs
from .interval_profile import DEFAULT_INTERVAL_PROFILE
from .multi_period import build_period_plan


def route_period_costs(*, route_length_km: float, peak_flow_pph: float,
                       vehicle_type: str, periods=DEFAULT_INTERVAL_PROFILE,
                       speed_kmh: float = 18.0, interval_reserve_sec: float = 20.0,
                       terminal_delay_reserve: float = .08,
                       charging_min_per_terminal: float = 10.0,
                       annual_days: int = 350, park_trip_coefficient: float = .90) -> dict:
    plan = build_period_plan(
        route_length_km=route_length_km, peak_flow_pph=peak_flow_pph,
        vehicle_type=vehicle_type, periods=periods, speed_kmh=speed_kmh,
        interval_reserve_sec=interval_reserve_sec,
        terminal_delay_reserve=terminal_delay_reserve,
        charging_min_per_terminal=charging_min_per_terminal,
        annual_days=annual_days, park_trip_coefficient=park_trip_coefficient,
    )
    v = plan[0] if plan else None
    peak_fleet = max((p.fleet for p in plan), default=0)
    annual_km = sum(p.annual_mileage_km for p in plan)
    annual_hours = sum(p.annual_hours for p in plan)
    costs = annual_route_costs(vehicle_type, annual_km, peak_fleet, annual_hours)
    return {"periods": [p.__dict__ if hasattr(p, "__dict__") else {k: getattr(p, k) for k in p.__slots__} for p in plan],
            "peak_fleet": peak_fleet, "annual_mileage_km": annual_km,
            "annual_hours": annual_hours, **costs}
