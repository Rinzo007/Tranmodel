"""Multi-period operating plan using the canonical daily interval profile."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod, validate_profile
from .vehicle_types import get_vehicle_type, round_down_half_minutes, round_up_to_interval

@dataclass(frozen=True, slots=True)
class PeriodPlan:
    period: str
    start: str
    end: str
    hours: float
    frequency_factor: float
    demand_pph: float
    frequency_vph: float
    interval_min: float
    cycle_time_min: float
    release: float
    fleet: int
    daily_trips: float
    annual_mileage_km: float
    annual_hours: float


def build_period_plan(*, route_length_km: float, peak_flow_pph: float, vehicle_type: str,
                      periods: tuple[IntervalPeriod, ...] = DEFAULT_INTERVAL_PROFILE,
                      speed_kmh: float = 18.0, interval_reserve_sec: float = 20.0,
                      terminal_delay_reserve: float = .08, charging_min_per_terminal: float = 10.0,
                      annual_days: int = 350, park_trip_coefficient: float = .90) -> list[PeriodPlan]:
    if route_length_km <= 0 or peak_flow_pph < 0 or speed_kmh <= 0:
        raise ValueError("Invalid route demand inputs")
    validate_profile(periods)
    v = get_vehicle_type(vehicle_type)
    peak_frequency = max(peak_flow_pph / v.capacity, 0.1)
    running_min = route_length_km / speed_kmh * 60.0
    out: list[PeriodPlan] = []
    for p in periods:
        freq = peak_frequency * p.frequency_factor
        raw_interval = 60.0 / freq
        interval = max(round_down_half_minutes(raw_interval + interval_reserve_sec / 60.0), .5)
        freq = 60.0 / interval
        cycle = running_min * (1.0 + terminal_delay_reserve)
        if v.charging_at_terminal:
            cycle += 2.0 * charging_min_per_terminal
        cycle = round_up_to_interval(cycle, interval)
        release = cycle / interval
        fleet = math.ceil(release / v.technical_readiness - 1e-9)
        daily_trips = p.hours * freq
        annual_trips = daily_trips * annual_days / park_trip_coefficient
        out.append(PeriodPlan(
            period=p.name, start=p.start, end=p.end, hours=p.hours,
            frequency_factor=p.frequency_factor, demand_pph=peak_flow_pph,
            frequency_vph=freq, interval_min=interval, cycle_time_min=cycle,
            release=release, fleet=fleet, daily_trips=daily_trips,
            annual_mileage_km=route_length_km * annual_trips,
            annual_hours=cycle / 60.0 * annual_trips,
        ))
    return out


def summarize_period_plan(plan: list[PeriodPlan], *, vehicle_type: str | None = None) -> dict:
    if not plan:
        return {"periods": [], "peak_fleet": 0, "daily_trips": 0.0, "annual_mileage_km": 0.0, "annual_hours": 0.0}
    result = {
        "periods": [asdict(p) for p in plan],
        "peak_fleet": max(p.fleet for p in plan),
        "daily_trips": sum(p.daily_trips for p in plan),
        "annual_mileage_km": sum(p.annual_mileage_km for p in plan),
        "annual_hours": sum(p.annual_hours for p in plan),
    }
    if vehicle_type is not None:
        v = get_vehicle_type(vehicle_type)
        peak_fleet = result["peak_fleet"]
        result.update({
            "vehicle_type": v.code,
            "vehicle_name": v.name,
            "capacity": v.capacity,
            "annual_contract_cost_mln": peak_fleet * v.annual_contract_cost_mln,
            "annual_amortization_mln": peak_fleet * v.annual_amortization_mln,
            "one_off_fleet_cost_mln": peak_fleet * v.one_off_cost_mln,
        })
    return result
