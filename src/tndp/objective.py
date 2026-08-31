"""Transparent multi-criteria objective and hard constraints for TNDP."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from math import isfinite
from .model import Evaluation, NetworkDesignConfig, RouteSet

@dataclass(frozen=True, slots=True)
class ObjectiveComponents:
    user_cost: float=0.0; operator_cost: float=0.0; uncovered_demand: float=0.0; transfers: float=0.0
    capacity_excess: float=0.0; duplication: float=0.0; walk_cost: float=0.0; wait_cost: float=0.0
    contract_cost: float=0.0; amortization: float=0.0; fuel_energy_cost: float=0.0; repair_cost: float=0.0
    crew_cost: float=0.0; infrastructure_cost: float=0.0; dispatch_cost: float=0.0

def combine_objective(c: ObjectiveComponents, config: NetworkDesignConfig)->float:
    return (c.user_cost*config.objective_user_time_weight + c.walk_cost*config.objective_walk_weight + c.wait_cost*config.objective_wait_weight + c.transfers*config.objective_transfer_weight + c.uncovered_demand*config.objective_uncovered_weight + c.capacity_excess*config.objective_overload_weight + c.operator_cost*config.objective_operating_weight + c.duplication*config.objective_route_duplication_weight + c.contract_cost*config.objective_contract_weight + c.amortization*config.objective_amortization_weight + (c.fuel_energy_cost+c.repair_cost+c.crew_cost+c.infrastructure_cost+c.dispatch_cost)*config.objective_operating_weight)

def evaluation_from_components(c: ObjectiveComponents, config: NetworkDesignConfig, *, metadata=None)->Evaluation:
    return Evaluation(score=combine_objective(c,config),user_cost=c.user_cost,operator_cost=c.operator_cost,uncovered_demand=c.uncovered_demand,transfers=c.transfers,capacity_excess=c.capacity_excess,metadata=metadata or {})

def apply_objective(route_set: RouteSet, evaluation: Evaluation, config: NetworkDesignConfig)->Evaluation:
    meta=dict(evaluation.metadata or {}); rc=meta.get("route_characteristics") or []
    def sum_route(key):
        return sum(float(x.get(key,0.0) or 0.0) for x in rc if isinstance(x,dict))
    contract=float(meta.get("annual_contract_cost_mln",sum_route("annual_contract_cost_mln")) or 0.0)
    amort=float(meta.get("annual_amortization_mln",sum_route("annual_amortization_mln")) or 0.0)
    fuel=float(meta.get("annual_fuel_energy_mln",sum_route("annual_fuel_energy_mln")) or 0.0)
    repair=float(meta.get("annual_repair_mln",sum_route("annual_repair_mln")) or 0.0)
    crew=float(meta.get("annual_crew_cost_mln",sum_route("annual_crew_cost_mln")) or 0.0)
    infra=float(meta.get("annual_infrastructure_mln",sum_route("annual_infrastructure_mln")) or 0.0)
    dispatch=float(meta.get("annual_dispatch_mln",sum_route("annual_dispatch_mln")) or 0.0)
    walk=float(meta.get("walking_cost",meta.get("walking_time",0.0)) or 0.0); wait=float(meta.get("waiting_cost",meta.get("waiting_time",0.0)) or 0.0)
    duplication=float(max(0,route_set.route_count()-len(route_set.unique_undirected_signatures())))
    c=ObjectiveComponents(max(0,float(evaluation.user_cost)),max(0,float(evaluation.operator_cost)),max(0,float(evaluation.uncovered_demand)),max(0,float(evaluation.transfers)),max(0,float(evaluation.capacity_excess)),duplication,max(0,walk),max(0,wait),max(0,contract),max(0,amort),max(0,fuel),max(0,repair),max(0,crew),max(0,infra),max(0,dispatch))
    violations=[]
    if route_set.route_count()<config.min_routes: violations.append(f"routes<{config.min_routes}")
    if route_set.route_count()>config.max_routes: violations.append(f"routes>{config.max_routes}")
    if evaluation.direct_demand_share<config.min_direct_demand_share: violations.append(f"direct_share<{config.min_direct_demand_share:.3f}")
    if evaluation.transfers>config.max_average_transfers: violations.append(f"average_transfers>{config.max_average_transfers:.3f}")
    if contract>config.max_annual_contract_cost_mln: violations.append(f"annual_contract_cost>{config.max_annual_contract_cost_mln:.3f}")
    fleet=int(float(meta.get("fleet",0) or 0));
    if fleet>config.max_fleet: violations.append(f"fleet>{config.max_fleet}")
    coverage=float(meta.get("coverage_share",1.0) or 0.0)
    if coverage<config.min_coverage_share: violations.append(f"coverage<{config.min_coverage_share:.3f}")
    for i,r in enumerate(route_set.routes):
        if r.frequency_vph<config.min_frequency_vph: violations.append(f"route[{i}].frequency<min")
        if r.frequency_vph>config.max_frequency_vph: violations.append(f"route[{i}].frequency>max")
        if len(r.nodes)<config.min_stops: violations.append(f"route[{i}].stops<min")
        if len(r.nodes)>config.max_stops: violations.append(f"route[{i}].stops>max")
    penalty=1e9+1e6*len(violations) if violations else 0.0; base=combine_objective(c,config); score=base+penalty
    if not isfinite(score): score=1e15; violations.append("non_finite_objective")
    meta.update({"objective_components":asdict(c),"objective_base_score":base,"objective_penalty":penalty,"feasible":not violations,"constraint_violations":violations,"annual_fuel_energy_mln":fuel,"annual_repair_mln":repair,"annual_crew_cost_mln":crew,"annual_infrastructure_mln":infra,"annual_dispatch_mln":dispatch,"annual_total_cost_mln":fuel+repair+crew+infra+dispatch+contract+amort})
    return Evaluation(score=float(score),user_cost=evaluation.user_cost,operator_cost=evaluation.operator_cost,uncovered_demand=evaluation.uncovered_demand,transfers=evaluation.transfers,direct_demand_share=evaluation.direct_demand_share,capacity_excess=evaluation.capacity_excess,metadata=meta)
