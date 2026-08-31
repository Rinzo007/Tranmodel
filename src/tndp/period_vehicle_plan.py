"""Build a unified daily/annual vehicle plan from six period assignments."""
from __future__ import annotations
from typing import Sequence
from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod
from .peak_fleet import reconcile_route_periods
from .cost_aggregation import aggregate_route_costs
from .route_loads import select_vehicle_for_route
from .vehicle_types import VEHICLE_TYPES


def build_route_vehicle_plan(*, route_id: str, route_length_km: float,
                             period_peak_flows: Sequence[float], vehicle_type: str,
                             periods: Sequence[IntervalPeriod] = DEFAULT_INTERVAL_PROFILE,
                             speed_kmh: float = 18.0, interval_reserve_sec: float = 20.0,
                             terminal_delay_reserve: float = .08,
                             charging_min_per_terminal: float = 10.0,
                             annual_days: int = 350, park_trip_coefficient: float = .90,
                             annual_contract_mln: float = 0.0,
                             annual_amortization_mln: float = 0.0) -> dict:
    if vehicle_type not in VEHICLE_TYPES:
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")
    rec = reconcile_route_periods(route_length_km=route_length_km, period_peak_flows=period_peak_flows,
                                  vehicle_type=vehicle_type, periods=periods, speed_kmh=speed_kmh,
                                  interval_reserve_sec=interval_reserve_sec,
                                  terminal_delay_reserve=terminal_delay_reserve,
                                  charging_min_per_terminal=charging_min_per_terminal,
                                  annual_days=annual_days, park_trip_coefficient=park_trip_coefficient)
    costs = aggregate_route_costs(vehicle_type=vehicle_type, annual_km=rec["annual_mileage_km"],
                                  fleet=rec["peak_fleet"], annual_hours=rec["annual_hours"],
                                  annual_contract_mln=annual_contract_mln,
                                  annual_amortization_mln=annual_amortization_mln)
    return {"route_id": str(route_id), "route_length_km": float(route_length_km),
            "vehicle_type": vehicle_type, "vehicle_name": rec["vehicle_name"],
            "peak_fleet": int(rec["peak_fleet"]), "annual_mileage_km": float(rec["annual_mileage_km"]),
            "annual_hours": float(rec["annual_hours"]), "periods": rec["periods"], "costs": costs}


def build_route_vehicle_plan_auto(*, route_id: str, route_length_km: float,
                                  period_peak_flows: Sequence[float],
                                  allowed_vehicle_types: Sequence[str], **kwargs) -> dict:
    """Choose the least-cost feasible vehicle against the worst assigned period flow."""
    peak_flow = max((float(x) for x in period_peak_flows), default=0.0)
    code, _ = select_vehicle_for_route(max_section_flow_pph=peak_flow, route_length_km=float(route_length_km),
                                       allowed_vehicle_types=allowed_vehicle_types,
                                       **{k: v for k, v in kwargs.items() if k in {
                                           "speed_kmh", "interval_reserve_sec", "terminal_delay_reserve",
                                           "charging_min_per_terminal", "annual_days", "park_trip_coefficient",
                                           "frequency_profile"}})
    return build_route_vehicle_plan(route_id=route_id, route_length_km=route_length_km,
                                    period_peak_flows=period_peak_flows, vehicle_type=code, **kwargs)


def build_network_vehicle_plan(route_specs: Sequence[dict], **kwargs) -> dict:
    defaults = dict(kwargs)
    plans = []
    for raw in route_specs:
        spec = dict(raw)
        auto = bool(spec.pop("auto_vehicle", False))
        allowed = spec.pop("allowed_vehicle_types", defaults.get("allowed_vehicle_types", tuple(VEHICLE_TYPES.keys())))
        if "vehicle_type" not in spec and not auto:
            spec["vehicle_type"] = defaults.get("default_vehicle_type", next(iter(allowed)))
        local = dict(defaults)
        local.pop("allowed_vehicle_types", None); local.pop("default_vehicle_type", None)
        local.update(spec)
        if auto:
            local.pop("vehicle_type", None)
            plans.append(build_route_vehicle_plan_auto(allowed_vehicle_types=allowed, **local))
        else:
            plans.append(build_route_vehicle_plan(**local))
    fleet = sum(int(p["peak_fleet"]) for p in plans)
    annual_km = sum(float(p["annual_mileage_km"]) for p in plans)
    annual_hours = sum(float(p["annual_hours"]) for p in plans)
    keys = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln", "total_annual_mln")
    costs = {k: sum(float(p["costs"].get(k, 0.0)) for p in plans) for k in keys}
    costs.update({"fleet": fleet, "annual_mileage_km": annual_km, "annual_hours": annual_hours,
                   "cost_per_km_rub": costs["total_annual_mln"] * 1_000_000 / max(annual_km, 1e-9)})
    return {"routes": plans, "fleet": fleet, "annual_mileage_km": annual_km, "annual_hours": annual_hours, "costs": costs}
