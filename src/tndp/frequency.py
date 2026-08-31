"""Peak and period frequency calculation for the TNDP operating plan."""
from __future__ import annotations
import math
from .interval_profile import DEFAULT_INTERVAL_PROFILE
from .model import NetworkDesignConfig, Route, RouteSet


def required_frequency_vph(max_hourly_load: float, config: NetworkDesignConfig, capacity: float | None = None) -> float:
    """Required peak departures/hour using capacity at 4 passengers/m²."""
    if max_hourly_load < 0: raise ValueError("max_hourly_load cannot be negative")
    cap = float(capacity if capacity is not None else config.vehicle_capacity)
    if cap <= 0: raise ValueError("capacity must be positive")
    return max(config.min_frequency_vph, min(config.max_frequency_vph, max_hourly_load / cap if max_hourly_load else config.min_frequency_vph))


def interval_from_frequency(frequency_vph: float, reserve_sec: float = 20.0) -> float:
    """Interval after adding reserve and rounding down to 0.5 min."""
    if frequency_vph <= 0: raise ValueError("frequency_vph must be positive")
    return max(.5, math.floor((60.0 / frequency_vph + reserve_sec / 60.0) * 2.0 + 1e-9) / 2.0)


def apply_frequency(route_set: RouteSet, route_peak_loads: dict[str, float], config: NetworkDesignConfig) -> RouteSet:
    """Adjust peak frequency from route loads while preserving vehicle and flow."""
    out = RouteSet()
    for route in route_set.routes:
        load = float(route_peak_loads.get(route.route_id or "", route.max_section_flow_pph))
        interval = interval_from_frequency(required_frequency_vph(load, config, route.capacity), config.interval_reserve_sec)
        out.add(Route(route.nodes, route.route_id, 60.0 / interval, load, route.vehicle_type))
    return out


def period_frequency(peak_frequency_vph: float, multiplier: float) -> float:
    return max(.1, float(peak_frequency_vph) * float(multiplier))


def default_profile_frequency_sum() -> float:
    return sum(p.hours * p.frequency_factor for p in DEFAULT_INTERVAL_PROFILE)
