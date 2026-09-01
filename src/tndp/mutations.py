"""Local-search mutations for TNDP route sets."""

from __future__ import annotations

import networkx as nx

from .model import NetworkDesignConfig, Route, RouteSet
from .vehicle_types import VEHICLE_TYPES


def _valid(route: Route, graph: nx.Graph, config: NetworkDesignConfig) -> bool:
    if not config.min_stops <= len(route.nodes) <= config.max_stops:
        return False
    try:
        length = nx.path_weight(graph, list(route.nodes), weight="length_km")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False
    return config.min_route_length_km <= float(length) <= config.max_route_length_km


def generate_mutations(route: Route, graph: nx.Graph, config: NetworkDesignConfig) -> list[Route]:
    """Generate bounded route, frequency and rolling-stock mutations."""
    out: list[Route] = []
    n = len(route.nodes)
    if n > config.min_stops:
        out.extend([route.with_nodes(route.nodes[1:]), route.with_nodes(route.nodes[:-1])])
    if n >= 3 and n - 1 >= config.min_stops:
        for i in range(1, n - 1):
            out.append(route.with_nodes(route.nodes[:i] + route.nodes[i + 1:]))

    for side in (0, 1):
        endpoint = route.nodes[0] if side == 0 else route.nodes[-1]
        neighbours = sorted(graph.neighbors(endpoint), key=lambda x: graph[endpoint][x].get("time", 0.0))
        for node in neighbours[: max(2, config.mutations_per_route // 4)]:
            if node in route.nodes:
                continue
            nodes = ((int(node),) + route.nodes) if side == 0 else (route.nodes + (int(node),))
            out.append(route.with_nodes(nodes))

    out.append(route.reversed())

    # Frequency mutations. They are deliberately modest because exact
    # assignment will decide whether the change actually improves the network.
    for factor in (0.75, 0.875, 1.125, 1.25):
        frequency = max(config.min_frequency_vph, min(config.max_frequency_vph, route.frequency_vph * factor))
        if abs(frequency - route.frequency_vph) > 1e-9:
            out.append(route.with_frequency(frequency))

    # Vehicle mutations are important because fleet composition is part of the
    # network-design decision, not merely a post-processing step.
    allowed = list(config.allowed_vehicle_types)
    current = route.vehicle_type
    if current in allowed:
        candidates = sorted(allowed, key=lambda code: abs(VEHICLE_TYPES[code].capacity - route.capacity))
        for code in candidates[: min(5, len(candidates))]:
            if code != current:
                out.append(route.with_vehicle_type(code))
    else:
        for code in allowed[: min(5, len(allowed))]:
            out.append(route.with_vehicle_type(code))

    unique: dict[tuple, Route] = {}
    for candidate in out:
        if candidate.nodes == route.nodes and candidate.frequency_vph == route.frequency_vph and candidate.vehicle_type == route.vehicle_type:
            continue
        # Geometry-changing mutations must remain feasible. Frequency/PС
        # mutations keep geometry and therefore need no graph traversal.
        if candidate.nodes != route.nodes and not _valid(candidate, graph, config):
            continue
        key = (candidate.nodes, round(candidate.frequency_vph, 6), candidate.vehicle_type)
        unique.setdefault(key, candidate)
        if len(unique) >= config.mutations_per_route * 2:
            break
    return list(unique.values())


def mutate_route_set(route_set: RouteSet, graph: nx.Graph, config: NetworkDesignConfig):
    """Yield route sets obtained by one local route/frequency/vehicle mutation."""
    for index, route in enumerate(route_set.routes):
        for replacement in generate_mutations(route, graph, config):
            trial = route_set.copy()
            trial.routes[index] = replacement
            if len({(r.nodes, round(r.frequency_vph, 6), r.vehicle_type) for r in trial.routes}) != trial.route_count():
                continue
            yield trial, {"operation": "mutate", "index": index, "route": replacement}

    if route_set.route_count() > config.min_routes:
        for index, route in enumerate(route_set.routes):
            trial = route_set.copy()
            trial.remove_at(index)
            yield trial, {"operation": "remove", "index": index, "route": route}
