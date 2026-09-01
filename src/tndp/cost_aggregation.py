"""Single source of truth for annual route and network cost aggregation."""
from __future__ import annotations
from .operating_costs import annual_route_costs

COST_COMPONENTS = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln")


def aggregate_route_costs(*, vehicle_type: str, annual_km: float, fleet: int, annual_hours: float,
                          route_length_km: float, annual_contract_mln: float = 0.0,
                          annual_amortization_mln: float = 0.0) -> dict[str, float]:
    if annual_km < 0 or annual_hours < 0 or fleet < 0 or route_length_km < 0:
        raise ValueError("annual_km, annual_hours, fleet and route_length_km must be non-negative")
    base = annual_route_costs(vehicle_type, annual_km, fleet, annual_hours, route_length_km)
    components = {key: float(base.get(key, 0.0)) for key in COST_COMPONENTS[:5]}
    components["contract_mln"] = float(annual_contract_mln)
    components["amortization_mln"] = float(annual_amortization_mln)
    total = sum(components.values())
    result = {**components, "fleet": int(fleet), "annual_km": float(annual_km),
              "annual_hours": float(annual_hours), "route_length_km": float(route_length_km),
              "total_annual_mln": total,
              "cost_per_km_rub": total * 1_000_000 / max(float(annual_km), 1e-9)}
    result["cost_share"] = {key: value / total if total > 0 else 0.0 for key, value in components.items()}
    # Keep a strict reconciliation invariant: no component may be counted twice.
    if abs(total - sum(result[k] for k in COST_COMPONENTS)) > 1e-9:
        raise RuntimeError("Annual route cost reconciliation failed")
    return result


def aggregate_network_costs(route_costs: list[dict]) -> dict[str, float]:
    result = {key: sum(float(r.get(key, 0.0)) for r in COST_COMPONENTS) for key in COST_COMPONENTS}
    result["total_annual_mln"] = sum(result.values())
    result["fleet"] = sum(int(r.get("fleet", 0)) for r in route_costs)
    result["annual_mileage_km"] = sum(float(r.get("annual_km", r.get("annual_mileage_km", 0.0))) for r in route_costs)
    result["annual_hours"] = sum(float(r.get("annual_hours", r.get("annual_in_service_hours", 0.0))) for r in route_costs)
    result["cost_per_km_rub"] = result["total_annual_mln"] * 1_000_000 / max(result["annual_mileage_km"], 1e-9)
    result["cost_share"] = {key: result[key] / result["total_annual_mln"] if result["total_annual_mln"] > 0 else 0.0 for key in COST_COMPONENTS}
    return result


def aggregate_peak_fleet(period_fleets: list[int | float]) -> int:
    return int(max((float(x) for x in period_fleets), default=0.0) + 0.999999)


def annualize_period_route_costs(period_route_costs: list[dict], *, peak_fleet: int) -> dict[str, float]:
    """Combine six-period mileage/hours while charging fleet-fixed costs once."""
    if not period_route_costs:
        return {"fleet": int(peak_fleet), "annual_km": 0.0, "annual_hours": 0.0, "total_annual_mln": 0.0}
    annual_km = sum(float(x.get("annual_mileage_km", x.get("annual_km", 0.0))) for x in period_route_costs)
    annual_hours = sum(float(x.get("annual_hours", x.get("annual_in_service_hours", 0.0))) for x in period_route_costs)
    first = period_route_costs[0]
    return aggregate_route_costs(
        vehicle_type=str(first["vehicle_type"]), annual_km=annual_km, fleet=int(peak_fleet),
        annual_hours=annual_hours, route_length_km=float(first.get("route_length_km", 0.0)),
        annual_contract_mln=float(first.get("annual_fleet_contract_cost_mln", first.get("annual_contract_mln", 0.0))),
        annual_amortization_mln=float(first.get("annual_fleet_amortization_mln", first.get("amortization_mln", 0.0))),
    )
