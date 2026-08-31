"""Generate feasible transit route candidates from demand corridors."""

from __future__ import annotations

import heapq
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
    """Create variants by adding high-demand nodes near the base path."""
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
    config: NetworkDesignConfig | None = None,
) -> list[Route]:
    """Generate a diverse route set from strongest OD corridors.

    The graph must contain edge attributes ``time`` (minutes) and
    ``length_km``. Node IDs are arbitrary; terminal constraints are handled by
    the caller or by pre-filtering candidate origins/destinations.
    """
    config = config or NetworkDesignConfig()
    if node_ids is None:
        node_ids = list(map(int, graph.nodes))
    node_to_idx = {int(n): i for i, n in enumerate(node_ids)}
    if demand_vector is None:
        demand_vector = np.zeros(len(node_ids), dtype=float)

    routes: list[Route] = []
    signatures: set[tuple[int, ...]] = set()
    for corridor in corridors:
        path, length_km = _shortest_path(graph, corridor.origin, corridor.destination)
        if not path:
            continue
        variants = _insert_high_demand_nodes(graph, path, demand_vector, node_to_idx, config)
        variants.append(list(reversed(path)))
        for order, variant in enumerate(variants):
            try:
                vlen = float(nx.path_weight(graph, variant, weight="length_km"))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if not _feasible(variant, vlen, config):
                continue
            sig = min(tuple(variant), tuple(reversed(variant)))
            if sig in signatures:
                continue
            signatures.add(sig)
            routes.append(
                Route(
                    nodes=tuple(variant),
                    route_id=f"cand_{len(routes)+1:05d}",
                    frequency_vph=6.0,
                )
            )
    return routes
