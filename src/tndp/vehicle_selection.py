"""Automatic rolling-stock selection for a route.

Selection is based on the supplied peak passenger flow and capacity at
4 passengers/m².  Alternatives are fully evaluated through the canonical
route-economics pipeline, so the caller can compare capacity, interval,
release, fleet and annual cost without duplicating formulas.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from .economics_core import calculate_annual_route_economics
from .vehicle_types import VEHICLE_TYPES


def _capacity_ok(capacity: float, flow: float) -> bool:
    return capacity >= flow


def evaluate_vehicle_alternatives(
    *,
    route_length_km: float,
    max_section_flow_pph: float,
    vehicle_codes: Iterable[str] | None = None,
    **economics_kwargs,
) -> list[dict]:
    """Evaluate every eligible vehicle and return alternatives by annual cost.

    All vehicles are retained in the result.  ``capacity_ok`` identifies
    vehicles whose 4 passengers/m² capacity can carry the input peak flow.
    ``feasible`` additionally requires a positive release and fleet.
    """
    codes = tuple(vehicle_codes or VEHICLE_TYPES.keys())
    results: list[dict] = []
    for code in codes:
        if code not in VEHICLE_TYPES:
            raise ValueError(f"Unknown vehicle type: {code}")
        vehicle = VEHICLE_TYPES[code]
        economics = calculate_annual_route_economics(
            vehicle_type=code,
            route_length_km=route_length_km,
            max_section_flow_pph=max_section_flow_pph,
            **economics_kwargs,
        )
        results.append({
            "vehicle_type": code,
            "vehicle_name": vehicle.name,
            "mode": vehicle.mode,
            "capacity_class": vehicle.capacity_class,
            "capacity": vehicle.capacity,
            "capacity_ok": _capacity_ok(vehicle.capacity, max_section_flow_pph),
            "feasible": bool(economics["release"] > 0 and economics["fleet"] > 0),
            "interval_min": economics["interval_min"],
            "frequency_vph": economics["frequency_vph"],
            "release": economics["release"],
            "fleet": economics["fleet"],
            "annual_mileage_km": economics["annual_mileage_km"],
            "annual_in_service_hours": economics["annual_in_service_hours"],
            "annual_total_cost_mln": economics["annual_total_cost_mln"],
            "cost_per_km_rub": economics["cost_per_km_rub"],
        })
    return results


def select_vehicle_type(
    *,
    route_length_km: float,
    max_section_flow_pph: float,
    vehicle_codes: Iterable[str] | None = None,
    objective: str = "cost",
    **economics_kwargs,
) -> dict:
    """Select the cheapest feasible vehicle, or the smallest-capacity one.

    ``objective='cost'`` minimizes annual cost among capacity-feasible vehicles.
    ``objective='capacity'`` minimizes capacity and uses annual cost as a tie
    breaker.  If no vehicle can carry the flow, the largest-capacity alternative
    is returned with ``capacity_ok=False`` so the caller can report unmet
    capacity rather than silently overstate service.
    """
    if max_section_flow_pph < 0:
        raise ValueError("max_section_flow_pph must be >= 0")
    if objective not in {"cost", "capacity"}:
        raise ValueError("objective must be 'cost' or 'capacity'")
    alternatives = evaluate_vehicle_alternatives(
        route_length_km=route_length_km,
        max_section_flow_pph=max_section_flow_pph,
        vehicle_codes=vehicle_codes,
        **economics_kwargs,
    )
    feasible = [x for x in alternatives if x["capacity_ok"] and x["feasible"]]
    if feasible:
        if objective == "capacity":
            selected = min(feasible, key=lambda x: (x["capacity"], x["annual_total_cost_mln"]))
        else:
            selected = min(feasible, key=lambda x: (x["annual_total_cost_mln"], x["capacity"]))
    else:
        selected = max(alternatives, key=lambda x: (x["capacity"], -x["annual_total_cost_mln"]))
        selected = {**selected, "capacity_warning": "No available vehicle has sufficient capacity"}
    return {"selected": selected, "alternatives": alternatives}
