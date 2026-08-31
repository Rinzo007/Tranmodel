"""Transport-zone construction and connectors for Tranmodel.

Zones are spatial demand units, independent from transit stops.  The default
scheme uses a regular projected grid clipped to the city boundary. Population
and attraction are aggregated from WorldPop cells/POIs already prepared by
phase 1.  Each zone also gets a centroid and a bounded set of nearby road
nodes that can be used as AequilibraE centroid connectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from config import CACHE_DIR, LAYERS_DIR, PROJ_EPSG, REPORT_DIR

DEFAULT_ZONE_SIZE_M = 750.0
DEFAULT_CONNECTORS = 3
DEFAULT_CONNECTOR_RADIUS_M = 1200.0


class ZoneError(RuntimeError):
    pass


def _load_boundary() -> gpd.GeoDataFrame:
    path = LAYERS_DIR / "boundary_voronezh.geojson"
    if not path.exists():
        path = CACHE_DIR / "boundary_voronezh.geojson"
    if not path.exists():
        raise FileNotFoundError("City boundary not found; run the boundary extraction first")
    return gpd.read_file(path).to_crs(PROJ_EPSG)


def _grid(boundary: gpd.GeoDataFrame, size_m: float) -> gpd.GeoDataFrame:
    geom = boundary.geometry.union_all()
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx, maxx + size_m, size_m)
    ys = np.arange(miny, maxy + size_m, size_m)
    polygons = []
    for x in xs[:-1]:
        for y in ys[:-1]:
            cell = box(x, y, x + size_m, y + size_m)
            if cell.intersects(geom):
                clipped = cell.intersection(geom)
                if not clipped.is_empty and clipped.area > 1000:
                    polygons.append(clipped)
    zones = gpd.GeoDataFrame({"geometry": polygons}, crs=PROJ_EPSG)
    zones.insert(0, "zone_id", np.arange(1, len(zones) + 1, dtype=np.int64))
    zones["centroid"] = zones.geometry.centroid
    zones["centroid_x"] = zones.centroid.x
    zones["centroid_y"] = zones.centroid.y
    zones["area_km2"] = zones.geometry.area / 1e6
    return zones


def _aggregate_population(zones: gpd.GeoDataFrame) -> pd.Series:
    cells_path = CACHE_DIR / "phase1_real" / "grid_cells.parquet"
    if not cells_path.exists():
        raise FileNotFoundError(f"WorldPop cells not found: {cells_path}")
    cells = gpd.read_parquet(cells_path).to_crs(PROJ_EPSG)
    cells = cells[["geometry", "pop"]].copy()
    cells["cell_point"] = cells.geometry.centroid
    joined = gpd.sjoin(
        gpd.GeoDataFrame(cells[["pop"]], geometry=cells["cell_point"], crs=PROJ_EPSG),
        zones[["zone_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    return joined.groupby("zone_id")["pop"].sum()


def _aggregate_jobs(zones: gpd.GeoDataFrame) -> pd.Series:
    path = CACHE_DIR / "phase1_real" / "pois_jobs.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    pois = gpd.read_parquet(path).to_crs(PROJ_EPSG)
    if "jobs" not in pois.columns:
        return pd.Series(dtype=float)
    joined = gpd.sjoin(pois[["jobs", "geometry"]], zones[["zone_id", "geometry"]],
                       how="inner", predicate="within")
    return joined.groupby("zone_id")["jobs"].sum()


def _road_connectors(zones: gpd.GeoDataFrame, roads: gpd.GeoDataFrame,
                     count: int, radius_m: float) -> dict[int, list[dict]]:
    """Return nearest road nodes for every zone centroid.

    Node identifiers are stable coordinate-derived strings; distances are in
    projected metres and can be used directly to create connector costs.
    """
    projected = roads.to_crs(PROJ_EPSG)
    coords: list[tuple[float, float]] = []
    for geom in projected.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            pts = list(line.coords)
            coords.extend((float(x), float(y)) for x, y, *_ in pts)
    unique = np.unique(np.asarray(coords, dtype=float), axis=0) if coords else np.empty((0, 2))
    if len(unique) == 0:
        raise ZoneError("Road layer has no usable vertices")

    from scipy.spatial import cKDTree
    tree = cKDTree(unique)
    out: dict[int, list[dict]] = {}
    centroid_xy = np.column_stack([zones.centroid_x, zones.centroid_y])
    distances, indices = tree.query(centroid_xy, k=min(count, len(unique)), distance_upper_bound=radius_m)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    for row_idx, zone_id in enumerate(zones.zone_id.to_numpy()):
        links = []
        for dist, idx in zip(distances[row_idx], indices[row_idx]):
            if not np.isfinite(dist) or idx >= len(unique):
                continue
            x, y = unique[int(idx)]
            links.append({
                "road_node": f"{x:.3f}_{y:.3f}",
                "x": float(x), "y": float(y), "distance_m": float(dist)
            })
        out[int(zone_id)] = links
    return out


def build_transport_zones(size_m: float = DEFAULT_ZONE_SIZE_M,
                          connectors: int = DEFAULT_CONNECTORS,
                          connector_radius_m: float = DEFAULT_CONNECTOR_RADIUS_M,
                          force: bool = False) -> gpd.GeoDataFrame:
    """Build and cache transport zones independent from transit stops."""
    out = CACHE_DIR / "zones"
    out.mkdir(parents=True, exist_ok=True)
    zones_path = out / "zones.parquet"
    report_path = out / "zones_report.json"
    if zones_path.exists() and report_path.exists() and not force:
        return gpd.read_parquet(zones_path)

    boundary = _load_boundary()
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    zones = _grid(boundary, size_m)
    if zones.empty:
        raise ZoneError("Zone generator returned no zones")

    pop = _aggregate_population(zones)
    jobs = _aggregate_jobs(zones)
    zones["population"] = zones["zone_id"].map(pop).fillna(0.0)
    zones["jobs"] = zones["zone_id"].map(jobs).fillna(0.0)
    zones["production"] = zones["population"]
    zones["attraction"] = zones["jobs"]

    connector_map = _road_connectors(zones, roads, connectors, connector_radius_m)
    zones["road_connectors"] = zones["zone_id"].map(connector_map)
    zones["n_connectors"] = zones["road_connectors"].map(len)
    zones.to_parquet(zones_path)

    report = {
        "zone_size_m": size_m,
        "n_zones": int(len(zones)),
        "population": float(zones.population.sum()),
        "jobs": float(zones.jobs.sum()),
        "connectors_per_zone": connectors,
        "zones_with_connectors": int((zones.n_connectors > 0).sum()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "zones_report.md").write_text(
        "\n".join([
            "# Транспортные зоны",
            "",
            f"- Размер ячейки: **{size_m:.0f} м**",
            f"- Зон: **{len(zones):,}**",
            f"- Население: **{zones.population.sum():,.0f}**",
            f"- Притяжение: **{zones.jobs.sum():,.0f}**",
            f"- Зон с коннекторами: **{(zones.n_connectors > 0).sum():,}**",
            "",
            "Зоны являются самостоятельными единицами спроса и не совпадают с остановками общественного транспорта.",
        ]), encoding="utf-8"
    )
    return zones


def zone_centroids(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a point GeoDataFrame with stable zone IDs."""
    return gpd.GeoDataFrame(
        zones[["zone_id", "population", "jobs", "production", "attraction"]].copy(),
        geometry=zones.geometry.centroid,
        crs=zones.crs,
    )


if __name__ == "__main__":
    result = build_transport_zones(force=True)
    print(result[["zone_id", "population", "jobs"]].head())
