"""Canonical physical route-operation model.

Units are explicit throughout this module:
- route length: km
- speed: km/h
- frequency: vehicles/hour
- interval and cycle time: minutes
- daily trips: vehicle trips/day
- annual mileage: km/year
- annual in-service hours: hours/year

The model follows the supplied Voronezh methodology: 18 km/h operating speed,
20 s interval reserve, 8% terminal delay reserve, optional 10 min charging at
each terminal, 350 operating days and a 0.90 park-trip coefficient.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Iterable

DEFAULT_FREQUENCY_PROFILE: tuple[tuple[float, float], ...] = (
    (1.0, 0.8),
    (2.0, 1.0),
    (7.5, 0.8),
    (3.0, 1.0),
    (1.5, 0.8),
    (3.0, 0.5),
)


@dataclass(frozen=True, slots=True)
class RouteOperatingCharacteristics:
    route_length_km: float
    max_section_flow_pph: float
    speed_kmh: float
    base_frequency_vph: float
    frequency_vph: float
    interval_min: float
    interval_reserve_sec: float
    terminal_delay_reserve: float
    charging_min_per_terminal: float
    turnaround_min: float
    release: int
    technical_readiness: float
    fleet: int
    daily_trips: float
    annual_mileage_km: float
    annual_in_service_hours: float


def _floor_half(value: float) -> float:
    """Round down to 0.5 minute."""
    return floor(value * 2.0 + 1e-9) / 2.0


def _ceil_to_interval(value: float, interval_min: float) -> float:
    return ceil(value / interval_min - 1e-9) * interval_min


def _validate_profile(profile: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    normalized = tuple((float(hours), float(factor)) for hours, factor in profile)
    if not normalized:
        raise ValueError("frequency_profile cannot be empty")
    if any(hours < 0 or factor <= 0 for hours, factor in normalized):
        raise ValueError("frequency profile must contain non-negative hours and positive factors")
    return normalized


def calculate_route_characteristics(
    route_length_km: float,
    max_section_flow_pph: float,
    *,
    capacity_at_4_ppm2: float = 73.0,
    speed_kmh: float = 18.0,
    interval_reserve_sec: float = 20.0,
    terminal_delay_reserve: float = 0.08,
    charging_min_per_terminal: float = 10.0,
    charging_at_terminal: bool = False,
    technical_readiness: float = 0.80,
    frequency_profile: Iterable[tuple[float, float]] | None = None,
    annual_days: int = 350,
    park_trip_coefficient: float = 0.90,
) -> RouteOperatingCharacteristics:
    length = float(route_length_km)
    flow = float(max_section_flow_pph)
    capacity = float(capacity_at_4_ppm2)
    speed = float(speed_kmh)
    reserve_sec = float(interval_reserve_sec)
    terminal_reserve = float(terminal_delay_reserve)
    charge_min = float(charging_min_per_terminal)

    if not all(isfinite(x) for x in (length, flow, capacity, speed, reserve_sec, terminal_reserve, charge_min)):
        raise ValueError("route operating parameters must be finite")
    if length <= 0:
        raise ValueError("route_length_km must be positive")
    if flow < 0:
        raise ValueError("max_section_flow_pph cannot be negative")
    if capacity <= 0 or speed <= 0:
        raise ValueError("capacity_at_4_ppm2 and speed_kmh must be positive")
    if reserve_sec < 0 or terminal_reserve < 0 or charge_min < 0:
        raise ValueError("reserve and charging times cannot be negative")
    if not 0 < technical_readiness <= 1:
        raise ValueError("technical_readiness must be in (0, 1]")
    if annual_days <= 0:
        raise ValueError("annual_days must be positive")
    if not 0 < park_trip_coefficient <= 1:
        raise ValueError("park_trip_coefficient must be in (0, 1]")

    # 4 passengers/m² capacity determines the base peak frequency.
    base_frequency = max(flow / capacity if flow > 0 else 0.1, 0.1)

    # The 20-second reserve is added before rounding down to 0.5 minute.
    interval_min = max(0.5, _floor_half(60.0 / base_frequency + reserve_sec / 60.0))
    frequency_from_interval = 60.0 / interval_min

    # Turnaround = running time × 1.08 + 10 min at each terminal for charging
    # when the selected vehicle requires terminal charging.
    running_min = length / speed * 60.0
    charging_total = 2.0 * charge_min if charging_at_terminal else 0.0
    turnaround_raw = running_min * (1.0 + terminal_reserve) + charging_total
    turnaround_min = _ceil_to_interval(turnaround_raw, interval_min)

    release = max(1, ceil(turnaround_min / interval_min - 1e-9))
    fleet = max(1, ceil(release / technical_readiness - 1e-9))

    profile = _validate_profile(frequency_profile or DEFAULT_FREQUENCY_PROFILE)
    daily_trips = sum(
        hours * factor * frequency_from_interval
        for hours, factor in profile
    )
    annual_mileage = length * daily_trips / park_trip_coefficient * annual_days
    annual_hours = turnaround_min * daily_trips / park_trip_coefficient * annual_days / 60.0

    return RouteOperatingCharacteristics(
        route_length_km=length,
        max_section_flow_pph=flow,
        speed_kmh=speed,
        base_frequency_vph=base_frequency,
        frequency_vph=frequency_from_interval,
        interval_min=interval_min,
        interval_reserve_sec=reserve_sec,
        terminal_delay_reserve=terminal_reserve,
        charging_min_per_terminal=charge_min if charging_at_terminal else 0.0,
        turnaround_min=turnaround_min,
        release=release,
        technical_readiness=float(technical_readiness),
        fleet=fleet,
        daily_trips=daily_trips,
        annual_mileage_km=annual_mileage,
        annual_in_service_hours=annual_hours,
    )
