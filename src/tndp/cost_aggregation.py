"""Unified annual route/network cost aggregation for TNDP."""
from __future__ import annotations
from .operating_costs import annual_route_costs


def aggregate_route_costs(*, vehicle_type: str, annual_km: float, fleet: int,
                          annual_hours: float, route_length_km: float,
                          annual_contract_mln: float = 0.0,
                          annual_amortization_mln: float = 0.0) -> dict[str, float]:
    base = annual_route_costs(vehicle_type, annual_km, fleet, annual_hours, route_length_km)
    total = base["total_before_vehicle"] + float(annual_contract_mln) + float(annual_amortization_mln)
    return {**base, "fleet": int(fleet), "annual_km": float(annual_km), "annual_hours": float(annual_hours),
            "contract_mln": float(annual_contract_mln), "amortization_mln": float(annual_amortization_mln),
            "total_annual_mln": total, "cost_per_km_rub": total * 1_000_000 / max(float(annual_km), 1e-9)}


def aggregate_network_costs(route_costs: list[dict]) -> dict[str, float]:
    keys = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln", "total_annual_mln")
    result = {k: sum(float(r.get(k, 0.0)) for r in route_costs) for k in keys}
    result["fleet"] = sum(int(r.get("fleet", 0)) for r in route_costs)
    result["annual_mileage_km"] = sum(float(r.get("annual_km", r.get("annual_mileage_km", 0.0))) for r in route_costs)
    result["annual_hours"] = sum(float(r.get("annual_hours", r.get("annual_in_service_hours", 0.0))) for r in route_costs)
    result["cost_per_km_rub"] = result["total_annual_mln"] * 1_000_000 / max(result["annual_mileage_km"], 1e-9)
    return result


def aggregate_peak_fleet(period_fleets: list[int | float]) -> int:
    return int(max((float(x) for x in period_fleets), default=0.0) + 0.999999)


def annualize_period_route_costs(period_route_costs: list[dict], *, peak_fleet: int) -> dict[str, float]:
    """Aggregate period mileage/hours while using a single physical peak fleet."""
    annual_km = sum(float(x.get("annual_mileage_km", x.get("annual_km", 0.0))) for x in period_route_costs)
    annual_hours = sum(float(x.get("annual_hours", x.get("annual_in_service_hours", 0.0))) for x in period_route_costs)
    if not period_route_costs:
        return {"fleet": int(peak_fleet), "annual_km": 0.0, "annual_hours": 0.0, "total_annual_mln": 0.0}
    first = period_route_costs[0]
    return aggregate_route_costs(
        vehicle_type=str(first.get("vehicle_type", "")), annual_km=annual_km,
        fleet=peak_fleet, annual_hours=annual_hours,
        route_length_km=float(first.get("route_length_km", 0.0)),
        annual_contract_mln=float(first.get("annual_contract_mln", 0.0)),
        annual_amortization_mln=float(first.get("amortization_mln", 0.0)),
    )
