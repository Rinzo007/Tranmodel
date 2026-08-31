"""Operating characteristics for a conditional transit route.

The calculations follow the project assumptions used for TNDP:
- 18 km/h commercial speed;
- 20 s interval reserve;
- interval rounded down to 0.5 min;
- 8% terminal delay reserve;
- 10 min electric-bus charging at each terminal, only when applicable;
- turnaround time rounded up to the operating interval;
- 80% technical readiness for buses and 90% for electric transit;
- annualization uses 0.9 park-trip coefficient and 350 days/year.
"""

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


def _floor_half(value: float) -> float:
    return floor(value * 2.0 + 1e-9) / 2.0


def _ceil_to_interval(value: float, interval_min: float) -> float:
    return ceil(value / interval_min - 1e-9) * interval_min


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
) -> RouteOperatingCharacteristics:
    """Calculate operating characteristics for one route.

    ``frequency_profile`` contains ``(hours, frequency_multiplier)`` pairs.
    Charging time is added at both terminals only for vehicles configured with
    ``charging_at_terminal=True`` (e.g. the terminal-charging electric bus).
    """
    length = float(route_length_km)
    flow = max(float(max_section_flow_pph), 0.0)
    capacity = float(capacity_at_4_ppm2)
    speed = float(speed_kmh)
    if length <= 0:
        raise ValueError("route_length_km must be positive")
    if capacity <= 0 or speed <= 0:
        raise ValueError("capacity_at_4_ppm2 and speed_kmh must be positive")
    if not 0 < technical_readiness <= 1:
        raise ValueError("technical_readiness must be in (0, 1]")

    frequency = max(flow / capacity if flow > 0 else 0.1, 0.1)
    raw_interval = 60.0 / frequency + interval_reserve_sec / 60.0
    interval = max(0.5, _floor_half(raw_interval))
    frequency_from_interval = 60.0 / interval

    running_min = length / speed * 60.0
    turnaround_raw = running_min * (1.0 + terminal_delay_reserve)
    effective_charging = charging_min_per_terminal if charging_at_terminal else 0.0
    turnaround_raw += 2.0 * effective_charging
    turnaround = _ceil_to_interval(turnaround_raw, interval)
    release = max(1, ceil(turnaround / interval - 1e-9))
    fleet = max(1, ceil(release / technical_readiness - 1e-9))

    profile = list(frequency_profile or (
        (3.0, 1.00),
        (6.0, 0.75),
        (4.0, 1.00),
        (3.0, 0.60),
        (8.0, 0.30),
    ))
    daily_trips = sum(max(0.0, hours) * max(0.0, multiplier) * frequency_from_interval for hours, multiplier in profile)
    annual_mileage = length * daily_trips / 0.9 * 350.0
    annual_hours = turnaround * daily_trips / 0.9 * 350.0 / 60.0

    return RouteOperatingCharacteristics(
        route_length_km=length,
        max_section_flow_pph=flow,
        speed_kmh=speed,
        frequency_vph=frequency_from_interval,
        interval_min=interval,
        terminal_delay_reserve=terminal_delay_reserve,
        charging_min_per_terminal=effective_charging,
        turnaround_min=turnaround,
        release=release,
        technical_readiness=technical_readiness,
        fleet=fleet,
        daily_trips=daily_trips,
        annual_mileage_km=annual_mileage,
        annual_in_service_hours=annual_hours,
    )
