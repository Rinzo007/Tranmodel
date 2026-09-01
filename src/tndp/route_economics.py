"""Single source of truth for physical route operating characteristics."""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable

@dataclass(frozen=True, slots=True)
class RouteOperatingCharacteristics:
    route_length_km: float
    max_section_flow_pph: float
    speed_kmh: float
    frequency_vph: float
    interval_min: float
    terminal_delay_reserve: float
    charging_min_per_terminal: float
    turnaround_min: float
    release: int
    technical_readiness: float
    fleet: int
    daily_trips: float
    annual_mileage_km: float
    annual_in_service_hours: float

def _floor_half(value: float) -> float: return floor(value * 2.0 + 1e-9) / 2.0
def _ceil_to_interval(value: float, interval_min: float) -> float: return ceil(value / interval_min - 1e-9) * interval_min

def calculate_route_characteristics(route_length_km: float, max_section_flow_pph: float, *, capacity_at_4_ppm2: float = 73.0, speed_kmh: float = 18.0, interval_reserve_sec: float = 20.0, terminal_delay_reserve: float = 0.08, charging_min_per_terminal: float = 10.0, charging_at_terminal: bool = False, technical_readiness: float = 0.80, frequency_profile: Iterable[tuple[float, float]] | None = None) -> RouteOperatingCharacteristics:
    length=float(route_length_km); flow=max(float(max_section_flow_pph),0.0); capacity=float(capacity_at_4_ppm2); speed=float(speed_kmh)
    if length<=0: raise ValueError("route_length_km must be positive")
    if capacity<=0 or speed<=0: raise ValueError("capacity_at_4_ppm2 and speed_kmh must be positive")
    if not 0<technical_readiness<=1: raise ValueError("technical_readiness must be in (0, 1]")
    frequency=max(flow/capacity if flow>0 else .1,.1)
    interval=max(.5,_floor_half(60/frequency+interval_reserve_sec/60))
    frequency_from_interval=60/interval
    running_min=length/speed*60
    turnaround_raw=running_min*(1+terminal_delay_reserve)+(2*charging_min_per_terminal if charging_at_terminal else 0)
    turnaround=_ceil_to_interval(turnaround_raw,interval)
    release=max(1,ceil(turnaround/interval-1e-9)); fleet=max(1,ceil(release/technical_readiness-1e-9))
    profile=list(frequency_profile or ((1.0,.8),(2.0,1.0),(7.5,.8),(3.0,1.0),(1.5,.8),(3.0,.5)))
    daily_trips=sum(max(0,h)*max(0,m)*frequency_from_interval for h,m in profile)
    annual_mileage=length*daily_trips/.9*350
    annual_hours=turnaround*daily_trips/.9*350/60
    return RouteOperatingCharacteristics(length,flow,speed,frequency_from_interval,interval,terminal_delay_reserve,(charging_min_per_terminal if charging_at_terminal else 0),turnaround,release,technical_readiness,fleet,daily_trips,annual_mileage,annual_hours)
