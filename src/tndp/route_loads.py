"""Route-segment passenger load reconstruction and fleet selection."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .interval_profile import DEFAULT_INTERVAL_PROFILE
from .model import RouteSet
from .vehicle_types import VEHICLE_TYPES, calculate_route_operations

@dataclass(frozen=True, slots=True)
class RouteLoad:
    route_index: int
    segment_loads_pph: tuple[float, ...]
    max_section_flow_pph: float
    max_section_index: int
    assigned_demand: float


def reconstruct_route_loads(route_set: RouteSet, demand: np.ndarray, *, stop_to_zone: dict[int, int], route_lengths_km: Sequence[float] | None = None, frequencies_vph: Sequence[float] | None = None) -> list[RouteLoad]:
    matrix=np.asarray(demand,dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: raise ValueError("demand must be a square OD matrix")
    if not route_set.routes: return []
    lengths=list(route_lengths_km or [1.0]*len(route_set.routes)); freqs=list(frequencies_vph or [r.frequency_vph for r in route_set.routes])
    pair_options={}
    for ri,route in enumerate(route_set.routes):
        positions={}
        for pos,stop in enumerate(route.nodes):
            zone=stop_to_zone.get(int(stop))
            if zone is not None: positions.setdefault(int(zone),[]).append(pos)
        for oz in positions:
            for dz in positions:
                if oz==dz: continue
                spans=[(a,b) for a in positions[oz] for b in positions[dz] if a<b]
                if spans:
                    a,b=min(spans,key=lambda x:x[1]-x[0]); pair_options.setdefault((oz,dz),[]).append((ri,a,b))
    loads=[np.zeros(len(r.nodes)-1,dtype=float) for r in route_set.routes]; assigned=[0.0]*len(route_set.routes)
    for (oz,dz),options in pair_options.items():
        q=float(matrix[oz,dz])
        if q<=0: continue
        scores=[]
        for ri,a,b in options:
            span=max(1,b-a); segment_min=max(1.0,float(lengths[ri])/max(1,len(route_set.routes[ri].nodes)-1)/18.0*60.0)
            scores.append((ri,a,b,max(float(freqs[ri]),.1)/(segment_min*span)))
        denominator=sum(x[3] for x in scores)
        if denominator<=0: continue
        for ri,a,b,attractiveness in scores:
            share=q*attractiveness/denominator; loads[ri][a:b]+=share; assigned[ri]+=share
    return [RouteLoad(ri,tuple(map(float,arr)),float(arr[int(np.argmax(arr))]) if arr.size else 0.0,int(np.argmax(arr)) if arr.size else -1,assigned[ri]) for ri,arr in enumerate(loads)]


def select_vehicle_for_route(*, max_section_flow_pph: float, route_length_km: float, allowed_vehicle_types: Sequence[str], speed_kmh: float=18.0, interval_reserve_sec: float=20.0, terminal_delay_reserve: float=.08, charging_min_per_terminal: float=10.0, annual_days: int=350, park_trip_coefficient: float=.90, frequency_profile=None) -> tuple[str, dict]:
    """Select the vehicle with the lowest full annual route cost."""
    profile = frequency_profile or tuple((p.hours,p.frequency_factor) for p in DEFAULT_INTERVAL_PROFILE)
    candidates=[]
    for code in allowed_vehicle_types:
        if code not in VEHICLE_TYPES: continue
        details=calculate_route_operations(route_length_km=route_length_km,max_section_flow_pph=max_section_flow_pph,vehicle_type=code,speed_kmh=speed_kmh,interval_reserve_sec=interval_reserve_sec,terminal_delay_reserve=terminal_delay_reserve,charging_min_per_terminal=charging_min_per_terminal,annual_days=annual_days,park_trip_coefficient=park_trip_coefficient,frequency_profile=profile)
        annual_cost=float(details["annual_total_operating_cost_mln"])
        candidates.append((annual_cost,float(details["interval_min"]),code,details))
    if not candidates: raise ValueError("No vehicle types available")
    _,_,code,details=min(candidates,key=lambda x:(x[0],x[1])); return code,details
