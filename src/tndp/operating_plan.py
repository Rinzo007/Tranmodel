"""Helpers for aggregating a network's route operating plans."""
from __future__ import annotations
from .multi_period import build_period_plan, summarize_period_plan

def build_network_operating_plan(route_specs, periods, **kwargs):
    """Build plans for route specs and aggregate annual network indicators.

    route_specs is an iterable of mappings containing route_length_km,
    peak_flow_pph and vehicle_type. The same route geometry may be used for
    every period while demand and service levels vary by period.
    """
    routes=[]
    for i, spec in enumerate(route_specs):
        plan=build_period_plan(route_length_km=float(spec['route_length_km']),
            peak_flow_pph=float(spec.get('peak_flow_pph',0.0)),
            vehicle_type=str(spec['vehicle_type']), periods=periods, **kwargs)
        summary=summarize_period_plan(plan)
        summary['route_id']=spec.get('route_id',str(i+1))
        routes.append(summary)
    return {
        'routes': routes,
        'peak_fleet': sum(r['peak_fleet'] for r in routes),
        'annual_mileage_km': sum(r['annual_mileage_km'] for r in routes),
        'annual_hours': sum(r['annual_hours'] for r in routes),
        'annual_contract_cost_mln': sum(r['annual_contract_cost_mln'] for r in routes),
        'annual_amortization_mln': sum(r['annual_amortization_mln'] for r in routes),
    }
