"""Real road-graph adapter used by TNDP candidate generation."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import scipy.spatial as spatial

from config import PROJ_EPSG

ROAD_SPEED_KMH = {
    "motorway": 90, "motorway_link": 50, "trunk": 70, "trunk_link": 40,
    "primary": 60, "primary_link": 35, "secondary": 50, "secondary_link": 30,
    "tertiary": 40, "tertiary_link": 25, "unclassified": 30, "residential": 30,
    "living_street": 20, "service": 20, "road": 30, "track": 15,
    "pedestrian": 5, "footway": 5, "cycleway": 15, "services": 20,
}


def _oneway_direction(value) -> int:
    if value is None:
        return 0
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return 1
    if text == "-1":
        return -1
    return 0


def build_tndp_graph(roads: gpd.GeoDataFrame) -> nx.DiGraph:
    """Build a directed OSM road graph with real length and travel-time weights."""
    graph = nx.DiGraph()
    projected = roads.to_crs(PROJ_EPSG).explode(index_parts=False, ignore_index=True)
    for _, row in projected.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        speed = float(ROAD_SPEED_KMH.get(str(row.get("highway") or "").lower(), 30.0))
        direction = _oneway_direction(row.get("oneway"))
        for line in lines:
            coords = list(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                if a == b:
                    continue
                u = (float(a[0]), float(a[1]))
                v = (float(b[0]), float(b[1]))
                length_km = float(np.hypot(a[0] - b[0], a[1] - b[1])) / 1000.0
                time_min = length_km / speed * 60.0
                attrs = {"time": time_min, "length_km": length_km}
                if direction == -1:
                    u, v = v, u
                if not graph.has_edge(u, v) or time_min < graph[u][v]["time"]:
                    graph.add_edge(u, v, **attrs)
                if direction == 0 and not graph.has_edge(v, u):
                    graph.add_edge(v, u, **attrs)
    return graph


def snap_stops_to_graph(graph: nx.Graph, stops: gpd.GeoDataFrame):
    """Snap transit stops to nearest road vertices."""
    projected = stops.to_crs(PROJ_EPSG).reset_index(drop=True)
    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("Road graph is empty")
    node_xy = np.asarray(nodes, dtype=float)
    tree = spatial.cKDTree(node_xy)
    stop_xy = np.column_stack([projected.geometry.x, projected.geometry.y])
    _, indices = tree.query(stop_xy, k=1)
    return graph, [nodes[int(i)] for i in indices], node_xy / 1000.0


def add_stop_nodes(graph: nx.Graph, stop_to_road_node: list[tuple[float, float]], k_neighbors: int = 8) -> nx.DiGraph:
    """Create a directed stop graph whose costs come from real-road shortest paths."""
    n = len(stop_to_road_node)
    out = nx.DiGraph()
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
        for raw_idx in np.atleast_1d(near):
            ui = int(raw_idx)
            if ui == stop_unique_index[stop_idx]:
                continue
            other = unique_nodes[ui]
            key = (road_node, other)
            if key not in shortest_cache:
                try:
                    path = nx.shortest_path(graph, road_node, other, weight="time")
                    shortest_cache[key] = (
                        float(nx.path_weight(graph, path, weight="time")),
                        float(nx.path_weight(graph, path, weight="length_km")),
                    )
                except nx.NetworkXNoPath:
                    continue
            time_min, length = shortest_cache[key]
            for j, node in enumerate(stop_to_road_node):
                if j == stop_idx or node != other:
                    continue
                current = out.get_edge_data(stop_idx, j)
                if current is None or time_min < current["time"]:
                    out.add_edge(stop_idx, j, time=time_min, length_km=length)
    return out
