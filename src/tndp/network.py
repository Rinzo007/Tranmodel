"""Adapters from Tranmodel OSM/stop layers to a TNDP graph."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.spatial as spatial
import geopandas as gpd

from config import PROJ_EPSG
from src.phase2 import ROAD_SPEED_KMH


def build_tndp_graph(roads: gpd.GeoDataFrame) -> nx.Graph:
    """Build an undirected road graph with time and length attributes."""
    graph = nx.Graph()
    projected = roads.to_crs(PROJ_EPSG)
    for _, row in projected.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        speed = float(ROAD_SPEED_KMH.get(str(row.get("highway") or "").lower(), 30.0))
        for line in lines:
            coords = list(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                if a == b:
                    continue
                length_km = float(np.hypot(a[0] - b[0], a[1] - b[1])) / 1000.0
                time_min = length_km / speed * 60.0
                attrs = {"time": time_min, "length_km": length_km}
                if graph.has_edge(a, b):
                    if time_min < graph[a][b]["time"]:
                        graph[a][b].update(attrs)
                else:
                    graph.add_edge(a, b, **attrs)
    return graph


def snap_stops_to_graph(
    graph: nx.Graph,
    stops: gpd.GeoDataFrame,
) -> tuple[nx.Graph, list[tuple[float, float]], np.ndarray]:
    """Snap stop points to nearest road vertices."""
    projected = stops.to_crs(PROJ_EPSG).reset_index(drop=True)
    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("Road graph is empty")
    node_xy = np.asarray(nodes, dtype=float)
    tree = spatial.cKDTree(node_xy)
    stop_xy_m = np.column_stack([projected.geometry.x, projected.geometry.y])
    _, idx = tree.query(stop_xy_m, k=1)
    mapping = [nodes[int(i)] for i in idx]
    return graph, mapping, node_xy / 1000.0


def add_stop_nodes(
    graph: nx.Graph,
    stop_to_road_node: list[tuple[float, float]],
    k_neighbors: int = 8,
) -> nx.Graph:
    """Create a compact stop graph using k-nearest neighbours.

    Exact road-network travel time is retained on each virtual stop-to-stop
    edge. Complexity is approximately O(n*k) shortest-path queries rather than
    O(n²) all-pairs stop construction.
    """
    n = len(stop_to_road_node)
    out = nx.Graph()
    out.add_nodes_from(range(n))
    if n < 2:
        return out

    unique_nodes = list(dict.fromkeys(stop_to_road_node))
    unique_xy = np.asarray(unique_nodes, dtype=float)
    tree = spatial.cKDTree(unique_xy)
    k = min(max(2, k_neighbors + 1), len(unique_nodes))
    unique_index = {node: i for i, node in enumerate(unique_nodes)}
    stop_unique_index = [unique_index[node] for node in stop_to_road_node]

    shortest_cache: dict[tuple[tuple[float, float], tuple[float, float]], tuple[float, float]] = {}
    for stop_idx, road_node in enumerate(stop_to_road_node):
        _, near = tree.query(road_node, k=k)
        near = np.atleast_1d(near)
        for raw_idx in near:
            ui = int(raw_idx)
            if ui == stop_unique_index[stop_idx]:
                continue
            other = unique_nodes[ui]
            key = tuple(sorted((road_node, other)))
            if key not in shortest_cache:
                try:
                    length = float(nx.path_weight(graph, nx.shortest_path(graph, key[0], key[1], weight="time"), weight="length_km"))
                    time_min = float(nx.shortest_path_length(graph, key[0], key[1], weight="time"))
                    shortest_cache[key] = (time_min, length)
                except nx.NetworkXNoPath:
                    continue
            time_min, length = shortest_cache[key]
            other_stops = [j for j, node in enumerate(stop_to_road_node) if node == other]
            for j in other_stops:
                if j == stop_idx:
                    continue
                current = out.get_edge_data(stop_idx, j)
                if current is None or time_min < current["time"]:
                    out.add_edge(stop_idx, j, time=time_min, length_km=length)
    return out
