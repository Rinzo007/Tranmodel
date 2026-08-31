"""Exports for generated TNDP route networks."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString

from config import PROJ_EPSG
from .model import RouteSet


def routes_to_geojson(route_set: RouteSet, stops: gpd.GeoDataFrame, path: str | Path) -> Path:
    """Write generated routes as WGS84 LineStrings with metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    projected = stops.to_crs(PROJ_EPSG).reset_index(drop=True)
    rows = []
    for route in route_set.routes:
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
