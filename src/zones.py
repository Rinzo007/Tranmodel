"""Transport-zone construction independent from transit stops."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import box

from config import CACHE_DIR, LAYERS_DIR, PROJ_EPSG, REPORT_DIR

DEFAULT_ZONE_SIZE_M = 750.0
DEFAULT_CONNECTORS = 3
DEFAULT_CONNECTOR_RADIUS_M = 1200.0


class ZoneError(RuntimeError):
    pass


def _load_boundary() -> gpd.GeoDataFrame:
    for path in (LAYERS_DIR / "boundary_voronezh.geojson", CACHE_DIR / "boundary_voronezh.geojson"):
        if path.exists():
            return gpd.read_file(path).to_crs(PROJ_EPSG)
    raise FileNotFoundError("City boundary not found; run boundary extraction first")


def _grid(boundary: gpd.GeoDataFrame, size_m: float) -> gpd.GeoDataFrame:
    geom = boundary.geometry.unary_union
    minx, miny, maxx, maxy = geom.bounds
    polygons = []
    for x in np.arange(minx, maxx, size_m):
        for y in np.arange(miny, maxy, size_m):
            cell = box(x, y, x + size_m, y + size_m)
            if cell.intersects(geom):
                clipped = cell.intersection(geom)
                if not clipped.is_empty and clipped.area > 1000:
                    polygons.append(clipped)
    zones = gpd.GeoDataFrame(
        {"zone_id": np.arange(1, len(polygons) + 1, dtype=np.int64)},
        geometry=polygons, crs=PROJ_EPSG,
    )
    centers = zones.geometry.centroid
    zones["centroid_x"] = centers.x
    zones["centroid_y"] = centers.y
    zones["area_km2"] = zones.geometry.area / 1e6
    return zones


def _aggregate_population(zones: gpd.GeoDataFrame) -> pd.Series:
    path = CACHE_DIR / "phase1_real" / "grid_cells.parquet"
    if not path.exists():
        raise FileNotFoundError(f"WorldPop cells not found: {path}")
    cells = gpd.read_parquet(path).to_crs(PROJ_EPSG)
    points = gpd.GeoDataFrame({"pop": cells["pop"]}, geometry=cells.geometry.centroid, crs=PROJ_EPSG)
    joined = gpd.sjoin(points, zones[["zone_id", "geometry"]], how="inner", predicate="within")
    return joined.groupby("zone_id")["pop"].sum()


def _aggregate_jobs(zones: gpd.GeoDataFrame) -> pd.Series:
    path = CACHE_DIR / "phase1_real" / "pois_jobs.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    pois = gpd.read_parquet(path).to_crs(PROJ_EPSG)
    if pois.empty or "jobs" not in pois.columns:
        return pd.Series(dtype=float)
    joined = gpd.sjoin(pois[["jobs", "geometry"]], zones[["zone_id", "geometry"]],
                       how="inner", predicate="within")
    return joined.groupby("zone_id")["jobs"].sum()


def _road_connectors(zones: gpd.GeoDataFrame, roads: gpd.GeoDataFrame,
                     count: int, radius_m: float) -> dict[int, list[dict]]:
    coords = []
    for geom in roads.to_crs(PROJ_EPSG).geometry:
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            coords.extend((float(x), float(y)) for x, y, *_ in line.coords)
    unique = np.unique(np.asarray(coords, dtype=float), axis=0) if coords else np.empty((0, 2))
    if len(unique) == 0:
        raise ZoneError("Road layer has no usable vertices")
    tree = cKDTree(unique)
    xy = zones[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    k = min(max(1, count), len(unique))
    distances, indices = tree.query(xy, k=k, distance_upper_bound=radius_m)
    distances, indices = np.atleast_2d(distances), np.atleast_2d(indices)
    if k == 1:
        distances, indices = distances.reshape(len(zones), 1), indices.reshape(len(zones), 1)
    result = {}
    for row_no, zone_id in enumerate(zones.zone_id.to_numpy()):
        result[int(zone_id)] = [
            {"x": float(unique[int(idx), 0]), "y": float(unique[int(idx), 1]), "distance_m": float(dist)}
            for dist, idx in zip(distances[row_no], indices[row_no])
            if np.isfinite(dist) and int(idx) < len(unique)
        ]
    return result


def build_transport_zones(size_m: float = DEFAULT_ZONE_SIZE_M,
                          connectors: int = DEFAULT_CONNECTORS,
                          connector_radius_m: float = DEFAULT_CONNECTOR_RADIUS_M,
                          force: bool = False) -> gpd.GeoDataFrame:
    """Build a polygon zoning system; stops are not used as zones."""
    out = CACHE_DIR / "zones"
    out.mkdir(parents=True, exist_ok=True)
    zones_path, report_path = out / "zones.parquet", out / "zones_report.json"
    if zones_path.exists() and report_path.exists() and not force:
        return gpd.read_parquet(zones_path)

    zones = _grid(_load_boundary(), size_m)
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    pop, jobs = _aggregate_population(zones), _aggregate_jobs(zones)
    zones["population"] = zones.zone_id.map(pop).fillna(0.0)
    zones["jobs"] = zones.zone_id.map(jobs).fillna(0.0)
    zones["production"] = zones["population"]
    zones["attraction"] = zones["jobs"]
    connector_map = _road_connectors(zones, roads, connectors, connector_radius_m)
    zones["n_connectors"] = zones.zone_id.map(lambda z: len(connector_map[int(z)]))
    zones["road_connectors_json"] = zones.zone_id.map(lambda z: json.dumps(connector_map[int(z)], separators=(",", ":")))
    zones.to_parquet(zones_path, index=False)

    report = {
        "zone_size_m": size_m,
        "n_zones": int(len(zones)),
        "population": float(zones.population.sum()),
        "jobs": float(zones.jobs.sum()),
        "connectors_per_zone": int(connectors),
        "zones_with_connectors": int((zones.n_connectors > 0).sum()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "zones_report.md").write_text("\n".join([
        "# Транспортные зоны", "",
        f"- Размер зоны: **{size_m:.0f} м**",
        f"- Зон: **{len(zones):,}**",
        f"- Население: **{zones.population.sum():,.0f}**",
        f"- Притяжение: **{zones.jobs.sum():,.0f}**",
        f"- Зон с дорожными коннекторами: **{(zones.n_connectors > 0).sum():,}**",
    ]), encoding="utf-8")
    return zones
