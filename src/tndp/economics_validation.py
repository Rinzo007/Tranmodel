"""Consistency checks for the authoritative route economics model."""
from __future__ import annotations

from math import isclose

from .economics_source import REQUIRED_COST_KEYS, calculate_annual_route_economics


def validate_route_economics(
    *,
    vehicle_type: str,
    route_length_km: float,
    max_section_flow_pph: float,
    **kwargs,
) -> dict:
    result = calculate_annual_route_economics(
        vehicle_type=vehicle_type,
        route_length_km=route_length_km,
        max_section_flow_pph=max_section_flow_pph,
        **kwargs,
    )
    costs = result["economics"]
    problems: list[str] = []

    # All monetary components are million rubles/year.
    for key in REQUIRED_COST_KEYS:
        if key not in costs:
            problems.append(f"missing_cost:{key}")
        elif float(costs[key]) < -1e-9:
            problems.append(f"negative_cost:{key}")

    component_sum = sum(float(costs.get(k, 0.0)) for k in REQUIRED_COST_KEYS[:-1])
    total = float(costs.get("total_annual_mln", 0.0))
    if not isclose(component_sum, total, rel_tol=1e-9, abs_tol=1e-8):
        problems.append("total_cost_mismatch")

    # Operational identities from the supplied methodology.
    fleet = int(result["fleet"])
    release = int(result["release"])
    ktg = float(result["technical_readiness"])
    if fleet < 1:
        problems.append("invalid_fleet")
    if release < 1:
        problems.append("invalid_release")
    if not (0 < ktg <= 1):
        problems.append("invalid_technical_readiness")
    elif fleet < int((release / ktg) - 1e-9):
        problems.append("fleet_below_ktg_requirement")

    annual_km = float(result["annual_mileage_km"])
    annual_hours = float(result["annual_in_service_hours"])
    daily_trips = float(result["daily_trips"])
    if daily_trips < 0 or annual_km < 0 or annual_hours < 0:
        problems.append("negative_operating_metric")

    days = int(kwargs.get("annual_days", 350))
    park_coeff = float(kwargs.get("park_trip_coefficient", 0.90))
    if days <= 0 or not 0 < park_coeff <= 1:
        problems.append("invalid_annualization_parameters")
    else:
        expected_km = float(result["route_length_km"]) * daily_trips / park_coeff * days
        expected_hours = float(result["turnaround_min"]) * daily_trips / park_coeff * days / 60.0
        if not isclose(annual_km, expected_km, rel_tol=1e-9, abs_tol=1e-7):
            problems.append("annual_mileage_formula_mismatch")
        if not isclose(annual_hours, expected_hours, rel_tol=1e-9, abs_tol=1e-7):
            problems.append("annual_hours_formula_mismatch")

    return {"ok": not problems, "problems": problems, "result": result}
