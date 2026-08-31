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
    """Generate bounded remove-node/extend/reverse mutations for one route."""
    out: list[Route] = []
    n = len(route.nodes)

    # Remove an endpoint or an interior stop.
    if n > config.min_stops:
        out.extend([
            route.with_nodes(route.nodes[1:]),
            route.with_nodes(route.nodes[:-1]),
        ])
    if n >= 3 and n - 1 >= config.min_stops:
        for i in range(1, n - 1):
            out.append(route.with_nodes(route.nodes[:i] + route.nodes[i + 1:]))

    # Extend either endpoint using nearby graph nodes.
    for side in (0, 1):
        endpoint = route.nodes[0] if side == 0 else route.nodes[-1]
        neighbours = sorted(
            graph.neighbors(endpoint),
            key=lambda x: graph[endpoint][x].get("time", 0.0),
        )
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
    """Yield route sets obtained by one local route mutation."""
    for index, route in enumerate(route_set.routes):
        for replacement in generate_mutations(route, graph, config):
            trial = route_set.copy()
            trial.routes[index] = replacement
            if len(trial.unique_undirected_signatures()) != trial.route_count():
                continue
            yield trial, {"operation": "mutate", "index": index, "route": replacement}

    # Remove whole routes when the lower route-count bound allows it.
    if route_set.route_count() > config.min_routes:
        for index, route in enumerate(route_set.routes):
            trial = route_set.copy()
            trial.remove_at(index)
            yield trial, {"operation": "remove", "index": index, "route": route}
