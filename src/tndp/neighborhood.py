"""Candidate-network neighborhood generation for TNDP local search."""
from __future__ import annotations

from .model import NetworkDesignConfig, Route, RouteSet


def _state_key(trial: RouteSet) -> tuple:
    return tuple(sorted((r.nodes, round(r.frequency_vph, 6), r.vehicle_type) for r in trial.routes))


def generate_network_moves(network: RouteSet, candidates: list[Route], config: NetworkDesignConfig):
    """Generate structural and service-plan moves without evaluating them.

    Moves are deliberately cheap. The optimizer applies the surrogate first and
    sends only the most promising states to the exact AequilibraE evaluator.
    """
    existing = {r.nodes for r in network.routes}
    seen = set()

    def emit(trial: RouteSet, meta: dict):
        key = _state_key(trial)
        if key in seen:
            return None
        seen.add(key)
        return trial, meta

    if network.route_count() < config.max_routes:
        for route in candidates:
            if route.nodes in existing:
                continue
            trial = network.copy()
            trial.add(route)
            value = emit(trial, {"operation": "add", "index": -1, "route": route})
            if value:
                yield value

    if network.route_count() > config.min_routes:
        for i, route in enumerate(network.routes):
            trial = network.copy()
            trial.remove_at(i)
            value = emit(trial, {"operation": "remove", "index": i, "route": route})
            if value:
                yield value

    # Replace a route with another candidate. Keep the raw neighborhood bounded;
    # exact evaluation remains the expensive step.
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
            value = emit(trial, {"operation": "swap", "index": i, "route": route})
            if value:
                yield value
                emitted += 1
            if emitted >= limit:
                break
        if emitted >= limit:
            break

    # Service-plan mutations: frequency and vehicle class are part of TNDP,
    # not merely post-processing. Use operationally meaningful frequency steps.
    frequency_steps = (-2.0, -1.0, 1.0, 2.0)
    for i, route in enumerate(network.routes):
        for delta in frequency_steps:
            f = min(config.max_frequency_vph, max(config.min_frequency_vph, route.frequency_vph + delta))
            if abs(f - route.frequency_vph) < 1e-9:
                continue
            trial = network.copy()
            trial.routes[i] = route.with_frequency(f)
            value = emit(trial, {"operation": "frequency", "index": i, "route": trial.routes[i], "delta_vph": f - route.frequency_vph})
            if value:
                yield value

        for vehicle_type in config.allowed_vehicle_types:
            if vehicle_type == route.vehicle_type:
                continue
            trial = network.copy()
            trial.routes[i] = route.with_vehicle_type(vehicle_type)
            value = emit(trial, {"operation": "vehicle", "index": i, "route": trial.routes[i], "vehicle_type": vehicle_type})
            if value:
                yield value
