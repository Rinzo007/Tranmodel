"""Build one coherent annual cost for a multi-period transit network."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence

from .cost_aggregation import aggregate_route_costs, aggregate_network_costs
from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod
from .peak_fleet import reconcile_route_periods

@dataclass(frozen=True, slots=True)
class RouteAnnualPlan:
    route_id: str
    vehicle_type: str
    peak_fleet: int
    annual_mileage_km: float
    annual_hours: float
    costs: dict
    period_operations: list[dict]


def build_route_annual_plan(*, route_id: str, route_length_km: float,
                            period_peak_flows: Sequence[float], vehicle_type: str,
                            annual_contract_mln: float = 0.0,
                            annual_amortization_mln: float = 0.0,
                            periods: Sequence[IntervalPeriod] = DEFAULT_INTERVAL_PROFILE,
                            **kwargs) -> RouteAnnualPlan:
    recon = reconcile_route_periods(route_length_km=route_length_km,
                                    period_peak_flows=period_peak_flows,
                                    vehicle_type=vehicle_type,
                                    periods=periods, **kwargs)
    costs = aggregate_route_costs(
        vehicle_type=vehicle_type,
        annual_km=recon["annual_mileage_km"],
        fleet=recon["peak_fleet"],
        annual_hours=recon["annual_hours"],
        annual_contract_mln=annual_contract_mln,
        annual_amortization_mln=annual_amortization_mln,
    )
    return RouteAnnualPlan(
        route_id=str(route_id), vehicle_type=vehicle_type,
        peak_fleet=int(recon["peak_fleet"]),
        annual_mileage_km=float(recon["annual_mileage_km"]),
        annual_hours=float(recon["annual_hours"]),
        costs=costs, period_operations=recon["periods"],
    )


def build_unified_network_cost(route_plans: Sequence[RouteAnnualPlan]) -> dict:
    rows = [dict(asdict(p).get("costs", {}), route_id=p.route_id, fleet=p.peak_fleet,
                 annual_mileage_km=p.annual_mileage_km, annual_hours=p.annual_hours)
            for p in route_plans]
    total = aggregate_network_costs(rows)
    total["route_count"] = len(route_plans)
    total["routes"] = [asdict(p) for p in route_plans]
    return total
