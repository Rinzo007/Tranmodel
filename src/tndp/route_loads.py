"""Route-segment passenger load reconstruction and fleet selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import Route, RouteSet
from .vehicle_types import VEHICLE_TYPES, calculate_route_operations


@dataclass(frozen=True, slots=True)
class RouteLoad:
    route_index: int
    segment_loads_pph: tuple[float, ...]
    max_section_flow_pph: float
    max_section_index: int
    assigned_demand: float


def reconstruct_route_loads(
    route_set: RouteSet,
    demand: np.ndarray,
    *,
    stop_to_zone: dict[int, int],
    route_lengths_km: Sequence[float] | None = None,
    frequencies_vph: Sequence[float] | None = None,
) -> list[RouteLoad]:
    """Estimate route loads from OD demand, splitting shared OD flows.

    This deterministic reconstruction is used between full AequilibraE
    assignments. AequilibraE remains the authoritative OD assignment; this
    function supplies the route/segment load needed for vehicle selection.
    """
    matrix = np.asarray(demand, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("demand must be a square OD matrix")
    if not route_set.routes:
        return []
    lengths = list(route_lengths_km or [1.0] * len(route_set.routes))
    freqs = list(frequencies_vph or [r.frequency_vph for r in route_set.routes])
    pair_options: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for ri, route in enumerate(route_set.routes):
        positions: dict[int, list[int]] = {}
        for pos, stop in enumerate(route.nodes):
            zone = stop_to_zone.get(int(stop))
            if zone is not None:
                positions.setdefault(int(zone), []).append(pos)
        zones = list(positions)
        for oz in zones:
            for dz in zones:
                if oz == dz:
                    continue
                spans = [(a, b) for a in positions[oz] for b in positions[dz] if a < b]
                if spans:
                    a, b = min(spans, key=lambda x: x[1] - x[0])
                    pair_options.setdefault((oz, dz), []).append((ri, a, b))
    loads = [np.zeros(len(r.nodes) - 1, dtype=float) for r in route_set.routes]
    assigned = [0.0] * len(route_set.routes)
    for (oz, dz), options in pair_options.items():
        q = float(matrix[oz, dz])
        if q <= 0:
            continue
        scores = []
        for ri, a, b in options:
            span = max(1, b - a)
            segment_min = max(1.0, float(lengths[ri]) / max(1, len(route_set.routes[ri].nodes) - 1) / 18.0 * 60.0)
            attractiveness = max(float(freqs[ri]), 0.1) / (segment_min * span)
            scores.append((ri, a, b, attractiveness))
        denominator = sum(x[3] for x in scores)
        if denominator <= 0:
            continue
        for ri, a, b, attractiveness in scores:
            share = q * attractiveness / denominator
            loads[ri][a:b] += share
            assigned[ri] += share
    result: list[RouteLoad] = []
    for ri, arr in enumerate(loads):
        if arr.size == 0:
            result.append(RouteLoad(ri, (), 0.0, -1, assigned[ri]))
        else:
            idx = int(np.argmax(arr))
            result.append(RouteLoad(ri, tuple(float(x) for x in arr), float(arr[idx]), idx, assigned[ri]))
    return result


def select_vehicle_for_route(
    *, max_section_flow_pph: float, route_length_km: float,
    allowed_vehicle_types: Sequence[str], speed_kmh: float = 18.0,
    interval_reserve_sec: float = 20.0, terminal_delay_reserve: float = 0.08,
    charging_min_per_terminal: float = 10.0, annual_days: int = 350,
    park_trip_coefficient: float = 0.90,
    frequency_profile=((3.0, 1.0), (6.0, 0.75), (4.0, 1.0), (3.0, 0.60), (8.0, 0.30)),
) -> tuple[str, dict]:
    """Select the least annualized-cost vehicle for the reconstructed flow."""
    candidates = []
    for code in allowed_vehicle_types:
        details = calculate_route_operations(
            route_length_km=route_length_km, max_section_flow_pph=max_section_flow_pph,
            vehicle_type=code, speed_kmh=speed_kmh, interval_reserve_sec=interval_reserve_sec,
            terminal_delay_reserve=terminal_delay_reserve, charging_min_per_terminal=charging_min_per_terminal,
            annual_days=annual_days, park_trip_coefficient=park_trip_coefficient,
            frequency_profile=frequency_profile,
        )
        annual_cost = float(details["annual_fleet_contract_cost_mln"] + details["annual_fleet_amortization_mln"])
        candidates.append((annual_cost, float(details["interval_min"]), code, details))
    if not candidates:
        raise ValueError("No vehicle types available")
    _, _, code, details = min(candidates, key=lambda x: (x[0], x[1]))
    return code, details
