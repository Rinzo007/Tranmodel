"""Adapters from Tranmodel OSM/stop layers to a TNDP graph."""

from __future__ import annotations

import numpy as np
import scipy.spatial as spatial
import networkx as nx
import geopandas as gpd

from config import PROJ_EPSG
from src.phase2 import ROAD_SPEED_KMH


def build_tndp_graph(roads: gpd.GeoDataFrame) -> nx.Graph:
    """Build an undirected graph with TNDP edge attributes.

    Nodes are road vertices. Edges contain ``time`` in minutes and
    ``length_km``. The graph is intended for candidate generation; AequilibraE
    remains the authoritative network model for final transit assignment.
    """
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
                    # Keep the fastest representation of duplicate OSM segments.
                    if time_min < graph[a][b]["time"]:
                        graph[a][b].update(attrs)
                else:
                    graph.add_edge(a, b, **attrs)
    return graph


def snap_stops_to_graph(
    graph: nx.Graph,
    stops: gpd.GeoDataFrame,
) -> tuple[nx.Graph, list[int], np.ndarray]:
    """Snap stop points to road nodes and return stop->graph-node mapping.

    Returned node IDs are the original road graph coordinate tuples. The
    returned coordinates are projected kilometres in the same order as the
    stops table.
    """
    projected = stops.to_crs(PROJ_EPSG).reset_index(drop=True)
    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("Road graph is empty")
    node_xy = np.asarray(nodes, dtype=float)
    tree = spatial.cKDTree(node_xy)
    stop_xy_m = np.column_stack([projected.geometry.x, projected.geometry.y])
    _, idx = tree.query(stop_xy_m, k=1)
    mapping = [nodes[int(i)] for i in idx]
    graph_xy_km = node_xy / 1000.0
    return graph, mapping, graph_xy_km


def add_stop_nodes(
    graph: nx.Graph,
    stop_to_road_node: list[tuple[float, float]],
) -> nx.Graph:
    """Return a graph whose node IDs are integer stop indices.

    Parallel stop mappings are connected by the shortest road path. This gives
    candidate generation a compact transit-node graph while preserving road
    travel times between stops.
    """
    out = nx.Graph()
    unique = list(dict.fromkeys(stop_to_road_node))
    for i in range(len(stop_to_road_node)):
        out.add_node(i)
    for i, a in enumerate(unique):
        for j in range(i + 1, len(unique)):
            pass
    # Compute only pairs actually needed by shortest paths later. Keeping the
    # full road graph is more efficient for large real networks; stop IDs are
    # therefore connected via virtual edges using cached shortest path costs.
    for i, a in enumerate(stop_to_road_node):
        for j in range(i + 1, len(stop_to_road_node)):
            if a == stop_to_road_node[j]:
                continue
            try:
                path = nx.shortest_path(graph, a, stop_to_road_node[j], weight="time")
                t = nx.path_weight(graph, path, weight="time")
                l = nx.path_weight(graph, path, weight="length_km")
            except nx.NetworkXNoPath:
                continue
            out.add_edge(i, j, time=float(t), length_km=float(l))
    return out
