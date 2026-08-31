"""Local-search mutations for TNDP route sets."""

from __future__ import annotations

import networkx as nx

from .model import NetworkDesignConfig, Route, RouteSet


def _valid(route: Route, graph: nx.Graph, config: NetworkDesignConfig) -> bool:
    if not config.min_stops <= len(route.nodes) <= config.max_stops:
        return False
    try:
        length = nx.path_weight(graph, list(route.nodes), weight="length_km")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False
    return config.min_route_length_km <= float(length) <= config.max_route_length_km


def generate_mutations(route: Route, graph: nx.Graph, config: NetworkDesignConfig) -> list[Route]:
    """Generate bounded remove/extend/shorten/reverse mutations for one route."""
    out: list[Route] = []
    n = len(route.nodes)

    if n > config.min_stops:
        out.extend([
            route.with_nodes(route.nodes[1:]),
            route.with_nodes(route.nodes[:-1]),
        ])

    if n >= 3:
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

    unique: dict[tuple[int, ...], Route] = {}
    for candidate in out:
        if candidate.nodes == route.nodes:
            continue
        if _valid(candidate, graph, config):
            sig = min(candidate.nodes, tuple(reversed(candidate.nodes)))
            unique.setdefault(sig, candidate)
        if len(unique) >= config.mutations_per_route:
            break
    return list(unique.values())


def mutate_route_set(route_set: RouteSet, graph: nx.Graph, config: NetworkDesignConfig):
    """Yield route sets obtained by one local mutation."""
    for index, route in enumerate(route_set.routes):
        for replacement in generate_mutations(route, graph, config):
            trial = route_set.copy()
            trial.routes[index] = replacement
            if len(trial.unique_undirected_signatures()) != trial.route_count():
                continue
            yield trial, index, replacement
