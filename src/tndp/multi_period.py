"""Multi-period operating plan derived from route peak demand.

The module keeps the OD matrix untouched and applies period factors to demand,
then recalculates frequency, interval, cycle time and fleet for every period.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from .model import Period
from .vehicle_types import get_vehicle_type, round_down_half_minutes, round_up_to_interval

@dataclass(frozen=True, slots=True)
class PeriodPlan:
    period: str
    hours: float
    demand_pph: float
    frequency_vph: float
    interval_min: float
    cycle_time_min: float
    release: float
    fleet: int
    daily_trips: float
    annual_mileage_km: float
    annual_hours: float
    annual_contract_cost_mln: float
    annual_amortization_mln: float


def build_period_plan(*, route_length_km: float, peak_flow_pph: float, vehicle_type: str,
                      periods: tuple[Period, ...], speed_kmh: float = 18.0,
                      interval_reserve_sec: float = 20.0, terminal_delay_reserve: float = .08,
                      charging_min_per_terminal: float = 10.0, annual_days: int = 350,
                      park_trip_coefficient: float = .90) -> list[PeriodPlan]:
    if route_length_km <= 0 or peak_flow_pph < 0: raise ValueError("Invalid route demand inputs")
    v = get_vehicle_type(vehicle_type)
    out=[]
    running = route_length_km / speed_kmh * 60.0
    for p in periods:
        demand = peak_flow_pph * p.demand_factor
        freq = max(demand / v.capacity, .1) * p.frequency_factor
        raw_interval = 60.0 / freq
        interval = max(round_down_half_minutes(raw_interval + interval_reserve_sec / 60.0), .5)
        freq = 60.0 / interval
        cycle = running * (1.0 + terminal_delay_reserve)
        if v.charging_at_terminal: cycle += 2.0 * charging_min_per_terminal
        cycle = round_up_to_interval(cycle, interval)
        release = cycle / interval
        fleet = int(__import__('math').ceil(release / v.technical_readiness - 1e-9))
        trips_day = p.hours * freq
        annual_trips = trips_day * annual_days / park_trip_coefficient
        out.append(PeriodPlan(p.name,p.hours,demand,freq,interval,cycle,release,fleet,trips_day,
            route_length_km*annual_trips,cycle/60.0*annual_trips,
            fleet*v.annual_contract_cost_mln*p.frequency_factor,
            fleet*v.annual_amortization_mln*p.frequency_factor))
    return out


def summarize_period_plan(plan: list[PeriodPlan]) -> dict:
    return {
        "periods": [asdict(p) for p in plan],
        "peak_fleet": max((p.fleet for p in plan), default=0),
        "annual_mileage_km": sum(p.annual_mileage_km for p in plan),
        "annual_hours": sum(p.annual_hours for p in plan),
        "annual_contract_cost_mln": sum(p.annual_contract_cost_mln for p in plan),
        "annual_amortization_mln": sum(p.annual_amortization_mln for p in plan),
    }
