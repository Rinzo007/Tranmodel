"""Reconstruct route-segment passenger loads from the zone OD matrix.

AequilibraE's transit skims provide OD-level results, while the current public
API does not expose a stable route-by-route load table. This module therefore
reconstructs route loads from the OD matrix using the routes present in the
candidate network. It is deliberately conservative and is used to update
vehicle choice/frequency between full assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .model import Route, RouteSet


@dataclass(frozen=True, slots=True)
class RouteLoad:
    route_index: int
    segment_loads_pph: tuple[float, ...]
    max_section_flow_pph: float
    max_section_index: int
    assigned_demand: float


def _route_stop_demand(route: Route, demand: np.ndarray, stop_to_zone: dict[int, int]) -> tuple[np.ndarray, float]:
    n = len(route.nodes) - 1
    loads = np.zeros(n, dtype=float)
    assigned = 0.0
    for oi, origin_stop in enumerate(route.nodes[:-1]):
        oz = stop_to_zone.get(int(origin_stop))
        if oz is None or oz >= demand.shape[0]:
            continue
        for di in range(oi + 1, len(route.nodes)):
            dest_stop = route.nodes[di]
            dz = stop_to_zone.get(int(dest_stop))
            if dz is None or dz >= demand.shape[1]:
                continue
            value = float(demand[oz, dz])
            if value <= 0:
                continue
            loads[oi:di] += value
            assigned += value
    return loads, assigned


def reconstruct_route_loads(
    route_set: RouteSet,
    demand: np.ndarray,
    *,
    stop_to_zone: dict[int, int],
    route_lengths_km: Sequence[float] | None = None,
    frequencies_vph: Sequence[float] | None = None,
) -> list[RouteLoad]:
    """Estimate peak passenger flow for every route segment.

    Demand is split among routes capable of carrying an OD pair in proportion
    to an attractiveness score (frequency divided by in-vehicle time). This
    avoids double-counting demand when several routes serve the same OD pair.
    """
    matrix = np.asarray(demand, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("demand must be a square OD matrix")
    if not route_set.routes:
        return []

    lengths = list(route_lengths_km or [1.0] * len(route_set.routes))
    freqs = list(frequencies_vph or [r.frequency_vph for r in route_set.routes])
    route_pairs: list[dict[tuple[int, int], tuple[int, int]]] = []
    pair_routes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for ri, route in enumerate(route_set.routes):
        pos = {int(stop): i for i, stop in enumerate(route.nodes)}
        pairs: dict[tuple[int, int], tuple[int, int]] = {}
        for a, ia in pos.items():
            za = stop_to_zone.get(a)
            if za is None:
                continue
            for b, ib in pos.items():
                zb = stop_to_zone.get(b)
                if zb is None or ia >= ib:
                    continue
                pairs[(za, zb)] = (ia, ib)
                pair_routes.setdefault((za, zb), []).append((ri, ib - ia))
        route_pairs.append(pairs)

    loads = [np.zeros(len(r.nodes) - 1, dtype=float) for r in route_set.routes]
    assigned_by_route = [0.0] * len(route_set.routes)
    for (oz, dz), options in pair_routes.items():
        q = float(matrix[oz, dz])
        if q <= 0:
            continue
        scores = []
        for ri, stop_span in options:
            travel_min = max(1.0, float(lengths[ri])) / 18.0 * 60.0 * max(1, stop_span) / max(1, len(route_set.routes[ri].nodes) - 1)
            score = max(float(freqs[ri]), 0.1) / travel_min
            scores.append((ri, stop_span, score))
        total_score = sum(x[2] for x in scores)
        if total_score <= 0:
            continue
        for ri, stop_span, score in scores:
            share = q * score / total_score
            ia, ib = route_pairs[ri][(oz, dz)]
            loads[ri][ia:ib] += share
            assigned_by_route[ri] += share

    result: list[RouteLoad] = []
    for ri, arr in enumerate(loads):
        if arr.size == 0:
            result.append(RouteLoad(ri, (), 0.0, -1, assigned_by_route[ri]))
            continue
        idx = int(np.argmax(arr))
        result.append(RouteLoad(ri, tuple(float(x) for x in arr), float(arr[idx]), idx, assigned_by_route[ri]))
    return result


def choose_vehicle_for_flow(
    max_section_flow_pph: float,
    *,
    vehicle_types: dict,
    route_length_km: float,
    evaluate_vehicle,
) -> tuple[str, dict]:
    """Choose the least-cost vehicle that can provide the required service."""
    candidates: list[tuple[float, str, dict]] = []
    for code in vehicle_types:
        result = evaluate_vehicle(code, route_length_km, max_section_flow_pph)
        if not math.isfinite(float(result["annual_cost"])):
            continue
        candidates.append((float(result["annual_cost"]), code, result))
    if not candidates:
        raise ValueError("No feasible vehicle type for route demand")
    cost, code, details = min(candidates, key=lambda x: x[0])
    return code, details
