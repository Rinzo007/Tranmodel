"""Candidate-network neighborhood generation for TNDP local search."""
from __future__ import annotations

from .model import NetworkDesignConfig, Route, RouteSet


def generate_network_moves(network: RouteSet, candidates: list[Route], config: NetworkDesignConfig):
    """Generate add/remove/swap moves without evaluating them.

    The optimizer applies the surrogate first and sends only the best moves to
    the exact AequilibraE evaluator.
    """
    existing = {r.nodes for r in network.routes}
    seen = set()

    if network.route_count() < config.max_routes:
        for route in candidates:
            if route.nodes in existing:
                continue
            trial = network.copy()
            trial.add(route)
            key = tuple((r.nodes, round(r.frequency_vph, 6), r.vehicle_type) for r in trial.routes)
            if key in seen:
                continue
            seen.add(key)
            yield trial, {"operation": "add", "index": -1, "route": route}

    if network.route_count() > config.min_routes:
        for i, route in enumerate(network.routes):
            trial = network.copy()
            trial.remove_at(i)
            key = tuple((r.nodes, round(r.frequency_vph, 6), r.vehicle_type) for r in trial.routes)
            if key in seen:
                continue
            seen.add(key)
            yield trial, {"operation": "remove", "index": i, "route": route}

    # Replace a route with a candidate. Limit the raw neighborhood; the
    # surrogate then selects the most promising exact evaluations.
    limit = max(20, config.mutations_per_route * 4)
    emitted = 0
    for i in range(network.route_count()):
        for route in candidates:
            if route.nodes in existing:
                continue
            trial = network.copy()
            trial.routes[i] = route
            if len({r.nodes for r in trial.routes}) != trial.route_count():
                continue
            key = tuple((r.nodes, round(r.frequency_vph, 6), r.vehicle_type) for r in trial.routes)
            if key in seen:
                continue
            seen.add(key)
            yield trial, {"operation": "swap", "index": i, "route": route}
            emitted += 1
            if emitted >= limit:
                return
