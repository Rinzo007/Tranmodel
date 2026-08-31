"""Generate feasible transit route candidates from demand corridors."""

from __future__ import annotations

from math import inf

import networkx as nx
import numpy as np

from .corridors import DemandCorridor
from .model import NetworkDesignConfig, Route


def _shortest_path(graph: nx.Graph, origin: int, destination: int) -> tuple[list[int], float]:
    try:
        path = nx.shortest_path(graph, origin, destination, weight="time")
        length = float(nx.path_weight(graph, path, weight="length_km"))
        return [int(x) for x in path], length
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], inf


def _feasible(path: list[int], length_km: float, config: NetworkDesignConfig) -> bool:
    return (
        config.min_stops <= len(path) <= config.max_stops
        and config.min_route_length_km <= length_km <= config.max_route_length_km
    )


def _insert_high_demand_nodes(
    graph: nx.Graph,
    base_path: list[int],
    demand_vector: np.ndarray,
    node_to_idx: dict[int, int],
    config: NetworkDesignConfig,
) -> list[list[int]]:
    """Create variants by inserting high-demand nodes into a shortest path."""
    variants = [base_path]
    if len(base_path) >= config.max_stops:
        return variants

    path_nodes = set(base_path)
    candidates = sorted(
        (n for n in graph.nodes if n not in path_nodes),
        key=lambda n: demand_vector[node_to_idx[n]] if n in node_to_idx else 0.0,
        reverse=True,
    )[: max(20, config.candidate_limit_per_corridor * 4)]

    for node in candidates:
        best_pos, best_delta = None, inf
        for pos in range(len(base_path) - 1):
            a, b = base_path[pos], base_path[pos + 1]
            try:
                old = graph[a][b].get("time", 0.0)
                p1 = nx.shortest_path(graph, a, node, weight="time")
                p2 = nx.shortest_path(graph, node, b, weight="time")
                new = nx.path_weight(graph, p1, weight="time") + nx.path_weight(graph, p2, weight="time")
                delta = float(new - old)
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos + 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if best_pos is not None:
            candidate = base_path[:best_pos] + [int(node)] + base_path[best_pos:]
            variants.append(candidate)
            if len(variants) >= config.candidate_limit_per_corridor:
                break
    return variants


def generate_route_candidates(
    corridors: list[DemandCorridor],
    graph: nx.Graph,
    node_xy_km: np.ndarray,
    node_ids: list[int] | None = None,
    demand_vector: np.ndarray | None = None,
    terminal_nodes: set[int] | None = None,
    config: NetworkDesignConfig | None = None,
) -> list[Route]:
    """Generate a diverse route set from strongest OD corridors.

    When ``terminal_nodes`` is supplied, every candidate starts and ends at a
    terminal. The OD corridor itself can originate/end elsewhere; the nearest
    permitted terminal is selected in projected space.
    """
    config = config or NetworkDesignConfig()
    if node_ids is None:
        node_ids = list(map(int, graph.nodes))
    node_to_idx = {int(n): i for i, n in enumerate(node_ids)}
    if demand_vector is None:
        demand_vector = np.zeros(len(node_ids), dtype=float)
    terminal_nodes = set(map(int, terminal_nodes or set()))

    if terminal_nodes:
        terminal_array = np.asarray(sorted(terminal_nodes), dtype=int)
        terminal_xy = node_xy_km[terminal_array]

    routes: list[Route] = []
    signatures: set[tuple[int, ...]] = set()
    for corridor in corridors:
        origin, destination = int(corridor.origin), int(corridor.destination)
        if terminal_nodes:
            if origin not in terminal_nodes:
                origin = int(terminal_array[np.argmin(((terminal_xy - node_xy_km[origin]) ** 2).sum(axis=1))])
            if destination not in terminal_nodes:
                destination = int(terminal_array[np.argmin(((terminal_xy - node_xy_km[destination]) ** 2).sum(axis=1))])
            if origin == destination:
                continue

        path, _ = _shortest_path(graph, origin, destination)
        if not path:
            continue
        variants = _insert_high_demand_nodes(graph, path, demand_vector, node_to_idx, config)
        variants.append(list(reversed(path)))
        for variant in variants:
            try:
                vlen = float(nx.path_weight(graph, variant, weight="length_km"))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if not _feasible(variant, vlen, config):
                continue
            if terminal_nodes and (variant[0] not in terminal_nodes or variant[-1] not in terminal_nodes):
                continue
            sig = min(tuple(variant), tuple(reversed(variant)))
            if sig in signatures:
                continue
            signatures.add(sig)
            routes.append(Route(nodes=tuple(variant), route_id=f"cand_{len(routes)+1:05d}", frequency_vph=6.0))
    return routes
