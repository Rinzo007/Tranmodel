"""Unified annual route cost aggregation for TNDP."""
from __future__ import annotations
from .operating_costs import annual_route_costs


def aggregate_route_costs(*, vehicle_type: str, annual_km: float, fleet: int,
                          annual_hours: float, annual_contract_mln: float = 0.0,
                          annual_amortization_mln: float = 0.0) -> dict[str, float]:
    """Combine mileage, crew, infrastructure, dispatching and fleet costs."""
    base = annual_route_costs(vehicle_type, annual_km, fleet, annual_hours)
    total = base["total_before_vehicle"] + float(annual_contract_mln) + float(annual_amortization_mln)
    return {**base, "fleet": int(fleet), "annual_km": float(annual_km), "annual_hours": float(annual_hours),
            "contract_mln": float(annual_contract_mln), "amortization_mln": float(annual_amortization_mln),
            "total_annual_mln": total,
            "cost_per_km_rub": total * 1_000_000 / max(float(annual_km), 1e-9)}


def aggregate_network_costs(route_costs: list[dict]) -> dict[str, float]:
    keys = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln", "total_annual_mln")
    result = {k: sum(float(r.get(k, 0.0)) for r in route_costs) for k in keys}
    # Fleet is a simultaneous resource: add route fleets as a conservative
    # upper bound; callers with interlined/corridor pooling can override it.
    result["fleet"] = sum(int(r.get("fleet", 0)) for r in route_costs)
    result["annual_mileage_km"] = sum(float(r.get("annual_km", r.get("annual_mileage_km", 0.0))) for r in route_costs)
    result["annual_hours"] = sum(float(r.get("annual_hours", r.get("annual_in_service_hours", 0.0))) for r in route_costs)
    result["cost_per_km_rub"] = result["total_annual_mln"] * 1_000_000 / max(result["annual_mileage_km"], 1e-9)
    return result


def aggregate_peak_fleet(period_fleets: list[int | float]) -> int:
    """Return simultaneous fleet requirement across time periods."""
    return int(max((float(x) for x in period_fleets), default=0.0) + 0.999999)
