"""Peak frequency calculation from maximum-section passenger flow."""
from __future__ import annotations
import math
from .model import NetworkDesignConfig, Route, RouteSet


def required_frequency_vph(max_hourly_load: float, config: NetworkDesignConfig, capacity: float | None = None) -> float:
    """Required peak departures/hour using capacity at 4 passengers/m²."""
    if max_hourly_load < 0:
        raise ValueError("max_hourly_load cannot be negative")
    cap = float(capacity if capacity is not None else config.vehicle_capacity)
    if cap <= 0:
        raise ValueError("capacity must be positive")
    return max(config.min_frequency_vph, min(config.max_frequency_vph, max_hourly_load / cap))


def apply_frequency(route_set: RouteSet, route_peak_loads: dict[str, float], config: NetworkDesignConfig) -> RouteSet:
    """Return a copy with frequencies adjusted from route peak loads.

    Vehicle type and maximum-section flow are preserved so subsequent
    operating-cost and fleet calculations use the selected rolling stock.
    """
    out = RouteSet()
    for route in route_set.routes:
        load = float(route_peak_loads.get(route.route_id or "", route.max_section_flow_pph))
        freq = math.ceil(required_frequency_vph(load, config, route.capacity) * 10.0) / 10.0
        out.add(Route(route.nodes, route.route_id, freq, load, route.vehicle_type))
    return out
