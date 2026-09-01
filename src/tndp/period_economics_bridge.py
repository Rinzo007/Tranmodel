"""Bridge period assignment results into one network-wide annual cost model."""
from __future__ import annotations
from typing import Sequence

from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod
from .period_vehicle_plan import build_route_vehicle_plan


def build_route_plan_from_period_evaluations(
    *, route_id: str, route_length_km: float, vehicle_type: str,
    period_evaluations: Sequence[dict],
    periods: Sequence[IntervalPeriod] = DEFAULT_INTERVAL_PROFILE,
    **kwargs,
) -> dict:
    """Extract route maximum-section flows from six assignment results.

    The expected structure is each period evaluation containing metadata with
    ``route_characteristics``. Missing periods are represented by zero flow.
    """
    period_peak_flows: list[float] = []
    for evaluation in period_evaluations:
        metadata = evaluation.get("metadata") or evaluation.get("evaluation", {}).get("metadata") or {}
        rows = metadata.get("route_characteristics") or []
        flow = 0.0
        for row in rows:
            if str(row.get("route_id", "")) == str(route_id):
                flow = float(row.get("max_section_flow_pph", 0.0) or 0.0)
                break
        period_peak_flows.append(flow)
    if len(period_peak_flows) != len(periods):
        period_peak_flows.extend([0.0] * (len(periods) - len(period_peak_flows)))
        period_peak_flows = period_peak_flows[:len(periods)]
    return build_route_vehicle_plan(
        route_id=route_id,
        route_length_km=route_length_km,
        period_peak_flows=period_peak_flows,
        vehicle_type=vehicle_type,
        periods=periods,
        **kwargs,
    )


def build_network_plan_from_period_evaluations(
    *, route_specs: Sequence[dict], period_results: Sequence[dict], periods=DEFAULT_INTERVAL_PROFILE, **kwargs,
) -> dict:
    """Build one consistent annual network plan from period assignments."""
    plans = []
    for spec in route_specs:
        plans.append(build_route_plan_from_period_evaluations(
            route_id=str(spec["route_id"]),
            route_length_km=float(spec["route_length_km"]),
            vehicle_type=str(spec["vehicle_type"]),
            period_evaluations=period_results,
            periods=periods,
            **kwargs,
        ))
    total = {
        "routes": plans,
        "fleet": sum(int(p["peak_fleet"]) for p in plans),
        "annual_mileage_km": sum(float(p["annual_mileage_km"]) for p in plans),
        "annual_hours": sum(float(p["annual_hours"]) for p in plans),
    }
    cost_keys = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln", "total_annual_mln")
    total["costs"] = {key: sum(float(p["costs"].get(key, 0.0)) for p in plans) for key in cost_keys}
    total["costs"]["fleet"] = total["fleet"]
    total["costs"]["annual_mileage_km"] = total["annual_mileage_km"]
    total["costs"]["annual_hours"] = total["annual_hours"]
    total["costs"]["cost_per_km_rub"] = total["costs"]["total_annual_mln"] * 1_000_000 / max(total["annual_mileage_km"], 1e-9)
    return total
