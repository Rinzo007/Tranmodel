"""Phase 1 (real data): demand assignment using stops from the reference file.

Stops come from `voronezh_routes_terminals.geojson` (NOT from OSM). All stops
participate and share a uniform walking-access radius (ACCESS_RADIUS_M).
Population (WorldPop cells) and jobs (POIs) are attached to the nearest stop
within that radius, mirroring Phase 1.

Output (data/cache/phase1_real/):
  - stops_demand.parquet   same schema as phase1 (osm_id = synthetic index id)
  - grid_cells.parquet     population cells (+ stop attachment)
  - pois_jobs.parquet      POIs with equal jobs share
  - phase1_report.json / .md
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config import CACHE_DIR, NAMES, PROJ_EPSG, REFERENCE_ROUTES_PATH, REPORT_DIR

JOBS_FACTOR = 0.5
ACCESS_RADIUS_M = 500.0


class Phase1RealError(RuntimeError):
    pass


def load_ref_stops() -> gpd.GeoDataFrame:
    """Load the repository-tracked reference stops and normalize to points."""
    ref_geojson = REFERENCE_ROUTES_PATH
    if not ref_geojson.exists():
        raise Phase1RealError(f"Reference file not found: {ref_geojson}")
    g = gpd.read_file(ref_geojson).to_crs(PROJ_EPSG)
    pts = g.geometry.centroid
    out = gpd.GeoDataFrame(
        {
            "osm_id": -(np.arange(len(g)) + 1),
            "kind": "stop",
            "name": g["name"],
            "is_terminal": g["is_terminal"],
            "n_routes": g["route_count"],
            "radius_m": ACCESS_RADIUS_M,
            "geometry": pts,
        },
        geometry="geometry",
        crs=PROJ_EPSG,
    )
    return out.reset_index(drop=True)


def load_cells_from_raster() -> gpd.GeoDataFrame:
    import rasterio
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as to_shape

    tif = CACHE_DIR / f"{NAMES['population']}.tif"
    with rasterio.open(tif) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    geoms, vals = [], []
    for geom, val in rio_shapes(arr, transform=transform):
        if val is None or val <= 0:
            continue
        geoms.append(to_shape(geom))
        vals.append(float(val))
    cells = gpd.GeoDataFrame({"pop": vals}, geometry=geoms, crs=crs).to_crs(PROJ_EPSG)
    cells["cx"] = cells.geometry.centroid.x
    cells["cy"] = cells.geometry.centroid.y
    cells["cell_id"] = np.arange(len(cells))
    return cells


def build_poi_jobs(pois, jobs_total):
    n = len(pois)
    if n == 0:
        raise Phase1RealError("No POIs provided for jobs distribution")
    share = jobs_total / n
    out = pois.copy()
    out["jobs"] = share
    return out


def assign_demand(stops, cells, pois_jobs):
    """Attach population/jobs to stops within access radius (nearest stop)."""
    import scipy.spatial as spatial

    radius = stops["radius_m"].to_numpy(dtype=float)
    stop_proj = stops.geometry.to_crs(PROJ_EPSG)
    stop_xy = np.column_stack([stop_proj.geometry.x, stop_proj.geometry.y])
    tree = spatial.cKDTree(stop_xy)

    cell_cx = cells["cx"].to_numpy()
    cell_cy = cells["cy"].to_numpy()
    cell_xy = np.column_stack([cell_cx, cell_cy])
    dist, idx = tree.query(cell_xy, k=1)
    cells = cells.copy()
    cells["stop_idx"] = np.where(dist <= radius[idx], idx, -1)
    cells["dist_m"] = dist
    pop_by_stop = cells.groupby("stop_idx")["pop"].sum().drop(labels=-1, errors="ignore")

    poi_xy = np.column_stack([pois_jobs.geometry.x, pois_jobs.geometry.y])
    d2, i2 = tree.query(poi_xy, k=1)
    stop_of_poi = np.where(d2 <= radius[i2], i2, -1)
    jobs_by_stop = pois_jobs.groupby(stop_of_poi)["jobs"].sum().drop(labels=-1, errors="ignore")

    res = stops.copy().reset_index(drop=True)
    res["population"] = res.index.map(pop_by_stop).fillna(0.0)
    res["jobs"] = res.index.map(jobs_by_stop).fillna(0.0)
    return res, cells


def run_phase1_real(force: bool = False) -> dict:
    out_dir = CACHE_DIR / "phase1_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stops_demand": out_dir / "stops_demand.parquet",
        "cells": out_dir / "grid_cells.parquet",
        "pois_jobs": out_dir / "pois_jobs.parquet",
        "report": out_dir / "phase1_report.json",
    }
    if all(p.exists() for p in paths.values()) and not force:
        return json.load(open(paths["report"], encoding="utf-8"))

    stops = load_ref_stops()
    print(f"Reference stops (from GeoJSON): {len(stops)}, radius={ACCESS_RADIUS_M} m")

    pop_summary = json.load(open(CACHE_DIR / f"{NAMES['population']}_summary.json", encoding="utf-8"))
    total_pop = pop_summary["total_population"]
    jobs_total = total_pop * JOBS_FACTOR

    cells = load_cells_from_raster()
    cells.to_crs(PROJ_EPSG).to_parquet(paths["cells"])

    pois = gpd.read_parquet(CACHE_DIR / "layers" / "pois.parquet").to_crs(PROJ_EPSG)
    pois_jobs = build_poi_jobs(pois, jobs_total)
    pois_jobs.to_parquet(paths["pois_jobs"])

    stops_demand, cells = assign_demand(stops, cells, pois_jobs)
    stops_demand.to_parquet(paths["stops_demand"])

    supplied_pop = float(stops_demand["population"].sum())
    supplied_jobs = float(stops_demand["jobs"].sum())
    covered_stops = int((stops_demand["population"] > 0).sum())

    report = {
        "total_population": round(total_pop, 1),
        "jobs_total": round(jobs_total, 1),
        "n_stops": int(len(stops_demand)),
        "n_cells": int(len(cells)),
        "n_pois": int(len(pois_jobs)),
        "access_radius_m": ACCESS_RADIUS_M,
        "population_sum_by_stop": round(supplied_pop, 1),
        "jobs_sum_by_stop": round(supplied_jobs, 1),
        "uncovered_population": round(float(total_pop - supplied_pop), 1),
        "uncovered_jobs": round(float(jobs_total - supplied_jobs), 1),
        "coverage_pop_share": round(supplied_pop / total_pop, 3) if total_pop else 0.0,
        "stops_with_population": covered_stops,
        "stops_with_jobs": int((stops_demand["jobs"] > 0).sum()),
        "source": "voronezh_routes_terminals.geojson",
    }
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _write_markdown(report)
    _write_map(stops_demand)
    return report


def _write_markdown(report: dict) -> None:
    lines = [
        "# Фаза 1 (реальные остановки) — Население и рабочие места",
        "",
        f"- Источник остановок: **{report['source']}**",
        f"- Остановок: {report['n_stops']}",
        f"- Радиус доступности: {report['access_radius_m']} м",
        f"- Население города: {report['total_population']:,.0f}",
        f"- Рабочие места (×0.5): {report['jobs_total']:,.0f}",
        "",
        "## Покрытие",
        f"- Население приписано: {report['population_sum_by_stop']:,.0f}"
        f" ({report['coverage_pop_share']*100:.1f}%)",
        f"- Рабочих мест приписано: {report['jobs_sum_by_stop']:,.0f}",
        f"- Остановок с населением: {report['stops_with_population']},",
        f" с работой: {report['stops_with_jobs']}",
        "",
    ]
    (REPORT_DIR / "phase1_real_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_map(stops_demand: gpd.GeoDataFrame) -> None:
    import folium
    stops_ll = stops_demand.to_crs("EPSG:4326")
    bbox = stops_ll.total_bounds
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    cmap = folium.LinearColormap(["#2c7bb6", "#ffffbf", "#d7191c"],
                                 vmin=0, vmax=max(stops_ll["population"].max(), 1e-6),
                                 caption="Население у остановки")
    for _, row in stops_ll.iterrows():
        pt = row.geometry.centroid
        folium.CircleMarker(
            location=[pt.y, pt.x], radius=5, color=cmap(row["population"]),
            fill=True, fillOpacity=0.6,
            popup=f"pop={row['population']:.0f}, jobs={row['jobs']:.0f}, routes={row['n_routes']}",
        ).add_to(m)
    cmap.add_to(m)
    m.save(str(REPORT_DIR / "phase1_real_map.html"))
    print(f"Map saved: {REPORT_DIR / 'phase1_real_map.html'}")


if __name__ == "__main__":
    report = run_phase1_real(force=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
