"""Consistency checks for the authoritative route economics adapter."""
from __future__ import annotations
from math import isclose
from .economics_source import REQUIRED_COST_KEYS, calculate_annual_route_economics


def validate_route_economics(*, vehicle_type: str, route_length_km: float,
                             max_section_flow_pph: float, **kwargs) -> dict:
    result = calculate_annual_route_economics(
        vehicle_type=vehicle_type,
        route_length_km=route_length_km,
        max_section_flow_pph=max_section_flow_pph,
        **kwargs,
    )
    costs = result["economics"]
    problems: list[str] = []
    for key in REQUIRED_COST_KEYS:
        if key not in costs:
            problems.append(f"missing_cost:{key}")
        elif float(costs[key]) < -1e-9:
            problems.append(f"negative_cost:{key}")
    component_sum = sum(float(costs.get(k, 0.0)) for k in REQUIRED_COST_KEYS[:-1])
    total = float(costs.get("total_annual_mln", 0.0))
    if not isclose(component_sum, total, rel_tol=1e-9, abs_tol=1e-8):
        problems.append("total_cost_mismatch")
    if result["fleet"] < 1:
        problems.append("invalid_fleet")
    if result["daily_trips"] < 0 or result["annual_mileage_km"] < 0 or result["annual_in_service_hours"] < 0:
        problems.append("negative_operating_metric")
    return {"ok": not problems, "problems": problems, "result": result}
