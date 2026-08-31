"""Exports for generated TNDP route networks."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

from config import PROJ_EPSG
from .model import RouteSet


def _path_geometry(graph: nx.Graph, a, b):
    path = nx.shortest_path(graph, a, b, weight="time")
    coords = [(float(x), float(y)) for x, y in path]
    return coords


def routes_to_geojson(route_set: RouteSet, stops: gpd.GeoDataFrame, path: str | Path,
                      road_graph: nx.Graph | None = None,
                      stop_to_road_node: list | None = None) -> Path:
    """Write routes as WGS84 LineStrings following the real road graph.

    If a road graph and stop-to-road-node mapping are supplied, each transit
    segment is represented by its actual shortest road path instead of a
    straight chord between stops.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    projected = stops.to_crs(PROJ_EPSG).reset_index(drop=True)
    rows = []
    for route in route_set.routes:
        if road_graph is not None and stop_to_road_node is not None:
            coords = []
            for a, b in zip(route.nodes[:-1], route.nodes[1:]):
                if not (0 <= int(a) < len(stop_to_road_node) and 0 <= int(b) < len(stop_to_road_node)):
                    continue
                try:
                    segment = _path_geometry(road_graph, stop_to_road_node[int(a)], stop_to_road_node[int(b)])
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    segment = []
                if segment:
                    if coords and coords[-1] == segment[0]:
                        coords.extend(segment[1:])
                    else:
                        coords.extend(segment)
        else:
            coords = []
            for node in route.nodes:
                idx = int(node)
                if 0 <= idx < len(projected):
                    point = projected.geometry.iloc[idx]
                    if point is not None and not point.is_empty:
                        coords.append((float(point.x), float(point.y)))
        if len(coords) >= 2:
            rows.append({
                "route_id": route.route_id or "",
                "frequency_vph": float(route.frequency_vph),
                "stop_count": len(route.nodes),
                "geometry": LineString(coords),
            })
    if not rows:
        raise ValueError("No valid route geometries to export")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=PROJ_EPSG).to_crs("EPSG:4326")
    gdf.to_file(target, driver="GeoJSON")
    return target
