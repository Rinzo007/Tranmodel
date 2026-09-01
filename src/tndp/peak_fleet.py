"""Peak fleet reconciliation across the six operating periods."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence

from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod
from .route_economics import calculate_route_characteristics
from .vehicle_types import get_vehicle_type

@dataclass(frozen=True, slots=True)
class PeriodRouteOperation:
    period_id: str
    period_name: str
    start: str
    end: str
    hours: float
    frequency_factor: float
    peak_flow_pph: float
    frequency_vph: float
    interval_min: float
    turnaround_min: float
    release: int
    fleet: int
    daily_trips: float
    annual_mileage_km: float
    annual_hours: float


def reconcile_route_periods(*, route_length_km: float, period_peak_flows: Sequence[float], vehicle_type: str,
                            periods: Sequence[IntervalPeriod] = DEFAULT_INTERVAL_PROFILE,
                            speed_kmh: float = 18.0, interval_reserve_sec: float = 20.0,
                            terminal_delay_reserve: float = .08, charging_min_per_terminal: float = 10.0,
                            annual_days: int = 350, park_trip_coefficient: float = .90) -> dict:
    """Use each period's assigned maximum section flow to derive a single fleet envelope.

    Frequency is recalculated independently for each period. The physical fleet
    is the maximum simultaneous requirement, while annual mileage/hours are
    accumulated from the period schedules.
    """
    if len(period_peak_flows) != len(periods):
        raise ValueError("period_peak_flows must match the period profile length")
    vehicle = get_vehicle_type(vehicle_type)
    rows = []
    peak_fleet = 0
    annual_km = 0.0
    annual_hours = 0.0
    for p, flow in zip(periods, period_peak_flows):
        multiplier = max(float(p.frequency_factor), .0)
        peak_flow = max(float(flow), 0.0)
        # A period frequency is the route's peak requirement scaled by the
        # profile. The operating calculator uses a single peak flow as the
        # planning envelope; we derive its interval for this period explicitly.
        op = calculate_route_characteristics(
            route_length_km=route_length_km,
            max_section_flow_pph=max(peak_flow / max(multiplier, .1), 0.0),
            capacity_at_4_ppm2=vehicle.capacity,
            speed_kmh=speed_kmh,
            interval_reserve_sec=interval_reserve_sec,
            terminal_delay_reserve=terminal_delay_reserve,
            charging_min_per_terminal=charging_min_per_terminal,
            charging_at_terminal=vehicle.charging_at_terminal,
            technical_readiness=vehicle.technical_readiness,
            frequency_profile=((p.hours, multiplier),),
        )
        fleet = op.fleet
        peak_fleet = max(peak_fleet, fleet)
        annual_km += op.annual_mileage_km
        annual_hours += op.annual_in_service_hours
        rows.append(PeriodRouteOperation(
            period_id=f"{p.number}_{p.start.replace(':','')}_{p.end.replace(':','')}",
            period_name=p.name, start=p.start, end=p.end, hours=p.hours,
            frequency_factor=multiplier, peak_flow_pph=peak_flow,
            frequency_vph=op.frequency_vph * multiplier,
            interval_min=60.0 / max(op.frequency_vph * multiplier, .1),
            turnaround_min=op.turnaround_min, release=op.release,
            fleet=fleet, daily_trips=op.daily_trips,
            annual_mileage_km=op.annual_mileage_km,
            annual_hours=op.annual_in_service_hours,
        ))
    return {
        "vehicle_type": vehicle_type,
        "vehicle_name": vehicle.name,
        "periods": [asdict(r) for r in rows],
        "peak_fleet": peak_fleet,
        "annual_mileage_km": annual_km,
        "annual_hours": annual_hours,
    }
