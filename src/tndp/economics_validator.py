"""Cross-module checks for route economics consistency."""
from __future__ import annotations


def validate_route_economics_record(record: dict, *, tolerance: float = 1e-6) -> list[str]:
    errors: list[str] = []
    costs = record.get("costs") or record.get("economic_breakdown") or {}
    parts = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln")
    for key in parts:
        value = float(costs.get(key, 0.0) or 0.0)
        if value < -tolerance:
            errors.append(f"negative_cost:{key}")
    total = float(costs.get("total_annual_mln", 0.0) or 0.0)
    expected = sum(float(costs.get(k, 0.0) or 0.0) for k in parts)
    if abs(total - expected) > tolerance * max(1.0, abs(expected)):
        errors.append("cost_total_mismatch")
    fleet = int(record.get("peak_fleet", record.get("fleet", 0)) or 0)
    annual_km = float(record.get("annual_mileage_km", record.get("annual_km", 0.0)) or 0.0)
    annual_hours = float(record.get("annual_hours", record.get("annual_in_service_hours", 0.0)) or 0.0)
    if fleet < 0: errors.append("negative_fleet")
    if annual_km < -tolerance: errors.append("negative_annual_km")
    if annual_hours < -tolerance: errors.append("negative_annual_hours")
    return errors


def validate_network_plan(plan: dict, *, tolerance: float = 1e-6) -> dict:
    errors: list[str] = []
    routes = plan.get("routes") or []
    for i, route in enumerate(routes):
        for error in validate_route_economics_record(route, tolerance=tolerance):
            errors.append(f"route[{i}]:{error}")
    network = plan.get("costs") or plan.get("network") or {}
    if network:
        parts = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln")
        expected = sum(float(network.get(k, 0.0) or 0.0) for k in parts)
        total = float(network.get("total_annual_mln", 0.0) or 0.0)
        if abs(total - expected) > tolerance * max(1.0, abs(expected)):
            errors.append("network_cost_total_mismatch")
        if int(network.get("fleet", plan.get("fleet", 0)) or 0) < 0:
            errors.append("network_negative_fleet")
    return {"valid": not errors, "errors": errors}
