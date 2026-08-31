"""Frequency setting from route loads and vehicle capacity."""

from __future__ import annotations

import math

from .model import NetworkDesignConfig, Route, RouteSet


def required_frequency_vph(max_hourly_load: float, config: NetworkDesignConfig) -> float:
    """Minimum departures/hour to keep the peak load below the target factor."""
    seats_per_hour = config.vehicle_capacity * config.target_load_factor
    return max(config.min_frequency_vph, min(config.max_frequency_vph, max_hourly_load / seats_per_hour))


def apply_frequency(route_set: RouteSet, route_peak_loads: dict[str, float], config: NetworkDesignConfig) -> RouteSet:
    """Return a copy with frequencies adjusted from peak route loads."""
    out = RouteSet()
    for route in route_set.routes:
        load = float(route_peak_loads.get(route.route_id or "", 0.0))
        freq = math.ceil(required_frequency_vph(load, config) * 10.0) / 10.0
        out.add(Route(route.nodes, route.route_id, freq))
    return out
