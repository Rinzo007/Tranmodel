"""Phase 1: population and jobs (attraction) grids, assignment to stops and POIs.

Inputs:
  - population_voronezh.tif        (WorldPop clipped raster)
  - layers/stops.parquet           (stops)
  - layers/pois.parquet            (POIs)

Outputs (data/cache/phase1/):
  - grid_cells.parquet             population/jobs per cell (cell_id, geometry, centroid)
  - stops_demand.parquet           stops + population_gen, jobs_attr (total & per POI)
  - pois_jobs.parquet              each POI with equal jobs share
  - phase1_report.json / .md       summary
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as to_shape

from config import (
    CACHE_DIR,
    LAYERS_DIR,
    NAMES,
    PROJ_EPSG,
    REPORT_DIR,
)

JOBS_FACTOR = 0.5  # jobs = population / 2

# Walking-access radius (metres) by OSM tag (STRICT — only these tags count).
#   railway=tram_stop   -> 600 m
#   highway=bus_stop    -> 500 m
#   railway=station     -> 1500 m
#   railway=halt        -> 1500 m
#   station=subway      -> 800 m
# Stops without any of these tags get NO radius and are excluded from demand
# assignment.
STOP_RADIUS_BY_TAG = [
    (("railway", "tram_stop"), 600.0),
    (("highway", "bus_stop"), 500.0),
    (("railway", "station"), 1500.0),
    (("railway", "halt"), 1500.0),
    (("station", "subway"), 800.0),
]


def stop_radius(stops: gpd.GeoDataFrame) -> pd.Series:
    """Radius (m) of the walking-access zone for each stop, strictly by tag.
    Stops matching none of the configured tags get NaN radius.
    """
    radius = pd.Series(np.nan, index=stops.index, dtype=float)
    for (key, value), r in STOP_RADIUS_BY_TAG:
        if key not in stops.columns:
            continue
        mask = stops[key].eq(value)
        radius[mask] = r
    return radius


class Phase1Error(RuntimeError):
    pass


def load_cells_from_raster() -> gpd.GeoDataFrame:
    """Rasterize WorldPop pixels into a GeoDataFrame of cells (WGS84)."""
    tif = CACHE_DIR / f"{NAMES['population']}.tif"
    with rasterio.open(tif) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    # Build polygons for pixels with population > 0
    geoms = []
    vals = []
    for g, val in rio_shapes(arr, transform=transform):
        if val is None or val <= 0:
            continue
        geoms.append(to_shape(g))
        vals.append(float(val))

    gdf = gpd.GeoDataFrame({"pop": vals}, geometry=geoms, crs=crs)
    # centroid for assignment
    gdf = gdf.to_crs(PROJ_EPSG)
    gdf["cx"] = gdf.geometry.centroid.x
    gdf["cy"] = gdf.geometry.centroid.y
    gdf["cell_id"] = np.arange(len(gdf))
    return gdf


def assign_cells_to_nearest(
    cells: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    target_col: str,
) -> gpd.GeoDataFrame:
    """Sum each target's value proportionally to nearest cell per centroid distance.

    Simplified nearest-neighbour (not proportional) assignment is fast and
    adequate at 100m resolution: each population cell keeps its value and maps
    to the closest stop; each POI's jobs map to the closest stop.
    """
    import scipy.spatial as spatial

    if cells.empty or targets.empty:
        raise Phase1Error("Empty cells or targets for assignment")

    cell_xy = cells[["cx", "cy"]].to_numpy(dtype=float)
    tgt_xy = targets.geometry.centroid.to_crs(PROJ_EPSG)
    tgt_arr = np.column_stack([tgt_xy.x, tgt_xy.y])

    tree = spatial.cKDTree(cell_xy)
    dist, idx = tree.query(tgt_arr, k=1)

    tgt = targets.copy()
    tgt["_cell_idx"] = idx
    # map cell total to whole stop demand later; here we keep the mapping
    cell_sums = (
        tgt.groupby("_cell_idx")[target_col]
        .sum()
        .rename(f"{target_col}_sum")
    )
    cells = cells.join(cell_sums)
    return cells


def build_poi_jobs(pois: gpd.GeoDataFrame, jobs_total: float) -> gpd.GeoDataFrame:
    """Distribute jobs equally across POIs (jobs_total / n_poi)."""
    n = len(pois)
    if n == 0:
        raise Phase1Error("No POIs provided for jobs distribution")
    share = jobs_total / n
    out = pois.copy()
    out["jobs"] = share
    return out


def build_stops_demand(
    stops: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    pois_jobs: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assign population (from cells) and jobs (from POIs) to the nearest stop
    whose walking-access radius covers the demand point. Only stops with a
    configured radius participate. Points outside any zone are unassigned.
    """
    import scipy.spatial as spatial

    radius = stop_radius(stops).to_numpy(dtype=float)
    has_radius = ~np.isnan(radius)
    radius_safe = radius[has_radius]

    stop_proj = stops.geometry[has_radius].to_crs(PROJ_EPSG)
    stop_pts = [g.centroid for g in stop_proj]
    stop_x = np.array([p.x for p in stop_pts])
    stop_y = np.array([p.y for p in stop_pts])
    stop_arr = np.column_stack([stop_x, stop_y])
    # index of participating stops within the full stops frame
    orig_idx = np.flatnonzero(has_radius)

    # 1) population cells -> stops within radius
    cell_proj = cells.geometry.to_crs(PROJ_EPSG)
    cell_ct = [g.centroid for g in cell_proj]
    cell_xy = np.column_stack([np.array([p.x for p in cell_ct]),
                               np.array([p.y for p in cell_ct])])
    tree = spatial.cKDTree(stop_arr)
    dist, idx = tree.query(cell_xy, k=1)
    cells = cells.copy()
    cells["stop_idx"] = np.where(dist <= radius_safe[idx], orig_idx[idx], -1)
    cells["dist_m"] = dist
    pop_by_stop = cells.groupby("stop_idx")["pop"].sum().drop(labels=-1, errors="ignore")

    # 2) POI jobs -> stops within radius
    pts = [g.centroid for g in pois_jobs.geometry.to_crs(PROJ_EPSG)]
    px = np.array([p.x for p in pts])
    py = np.array([p.y for p in pts])
    poi_arr = np.column_stack([px, py])
    d2, i2 = tree.query(poi_arr, k=1)
    stop_of_poi = np.where(d2 <= radius_safe[i2], orig_idx[i2], -1)
    jobs_by_stop = pois_jobs.groupby(stop_of_poi)["jobs"].sum().drop(labels=-1, errors="ignore")

    res = stops.copy().reset_index(drop=True)
    res["radius_m"] = radius
    res["population"] = res.index.map(pop_by_stop).fillna(0.0)
    res["jobs"] = res.index.map(jobs_by_stop).fillna(0.0)
    return res, cells


def run_phase1(force: bool = False) -> dict:
    out_dir = CACHE_DIR / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    coll = {
        "cells": out_dir / "grid_cells.parquet",
        "stops_demand": out_dir / "stops_demand.parquet",
        "pois_jobs": out_dir / "pois_jobs.parquet",
    }
    if all(p.exists() for p in coll.values()) and not force:
        print("Phase 1 loaded from cache")
        report_path = out_dir / "phase1_report.json"
        if report_path.exists():
            return json.load(open(report_path, encoding="utf-8"))

    # --- load base layers ---
    stops = gpd.read_parquet(LAYERS_DIR / "stops.parquet")
    pois = gpd.read_parquet(LAYERS_DIR / "pois.parquet")
    pop_summary = json.load(open(CACHE_DIR / f"{NAMES['population']}_summary.json", encoding="utf-8"))
    total_pop = pop_summary["total_population"]
    jobs_total = total_pop * JOBS_FACTOR

    print(f"Total population: {total_pop:,.0f} -> jobs: {jobs_total:,.0f} (x{JOBS_FACTOR})")
    print(f"Stops: {len(stops)}, POIs: {len(pois)}")

    # --- 1) cells ---
    print("Building population/jobs cells from raster ...")
    cells = load_cells_from_raster()
    cells.to_crs(PROJ_EPSG).to_parquet(coll["cells"])
    print(f"  cells: {len(cells)}")

    # --- 2) POI jobs (equal share) ---
    print("Distributing jobs across POIs ...")
    pois_jobs = build_poi_jobs(pois, jobs_total)
    print(f"  per-POI jobs: {pois_jobs['jobs'].iloc[0]:.2f}")
    pois_jobs.to_parquet(coll["pois_jobs"])

    # --- 3) stops demand ---
    print("Assigning population & jobs to stops within access radius ...")
    stops_demand, cells = build_stops_demand(stops, cells, pois_jobs)
    stops_demand.to_parquet(coll["stops_demand"])

    # coverage outside any zone
    uncovered_pop = float(cells.loc[cells["stop_idx"] == -1, "pop"].sum())
    supplied_pop = float(stops_demand["population"].sum())
    uncovered_jobs = float(jobs_total - stops_demand["jobs"].sum())

    print(f"  stops covered: {(stops_demand['population']>0).sum()}, "
          f"uncovered population: {uncovered_pop:,.0f}")
    print("  radius distribution (participating stops only):")
    print(stops_demand.loc[stops_demand["radius_m"].notna()].groupby("radius_m")["population"].agg(["mean", "size"]).to_string())
    print(f"  stops WITHOUT configured radius (excluded): {stops_demand['radius_m'].isna().sum()}")

    report = {
        "total_population": round(total_pop, 1),
        "jobs_total": round(jobs_total, 1),
        "n_cells": int(len(cells)),
        "n_stops": int(len(stops_demand)),
        "n_pois": int(len(pois_jobs)),
        "jobs_per_poi": round(float(pois_jobs["jobs"].iloc[0]), 3),
        "population_sum_by_stop": round(supplied_pop, 1),
        "jobs_sum_by_stop": round(float(stops_demand["jobs"].sum()), 1),
        "uncovered_population": round(uncovered_pop, 1),
        "uncovered_jobs": round(uncovered_jobs, 1),
        "coverage_pop_share": round(supplied_pop / total_pop, 3),
        "stops_with_population": int((stops_demand["population"] > 0).sum()),
        "stops_with_jobs": int((stops_demand["jobs"] > 0).sum()),
        "stops_radius_distribution": {
            str(int(r)): int(c) for r, c in stops_demand.loc[
                stops_demand["radius_m"].notna()
            ]["radius_m"].value_counts().items()
        },
        "stops_without_radius": int(stops_demand["radius_m"].isna().sum()),
    }
    with open(out_dir / "phase1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _write_markdown(report)
    _write_map(stops_demand, pois_jobs)
    return report


def _write_markdown(report: dict) -> None:
    lines = [
        "# Фаза 1 — Население и рабочие места",
        "",
        f"- Население города: **{report['total_population']:,.0f}**",
        f"- Рабочие места (×0.5): **{report['jobs_total']:,.0f}**",
        f"- Ячеек сетки: {report['n_cells']:,}",
        f"- Остановок: {report['n_stops']}",
        f"- POI: {report['n_pois']}",
        f"- Рабочих мест на POI: {report['jobs_per_poi']}",
        "",
        "## Распределение по зонам доступности (радиусы)",
        "",
    ]
    for r, c in report.get("stops_radius_distribution", {}).items():
        lines.append(f"- Радиус **{r} м**: {c} остановок")
    lines += [
        f"- Остановок без радиуса (не участвуют): {report.get('stops_without_radius', 0)}",
        "",
        "## Покрытие",
        f"- Население приписано к остановкам: {report['population_sum_by_stop']:,.0f}"
        f" ({report['coverage_pop_share']*100:.1f}%)",
        f"- Рабочих мест приписано: {report['jobs_sum_by_stop']:,.0f}",
        f"- Вне зон доступности — население: {report['uncovered_population']:,.0f},"
        f" работы: {report['uncovered_jobs']:,.0f}",
        f"- Остановок с населением: {report['stops_with_population']},"
        f" с работой: {report['stops_with_jobs']}",
        "",
    ]
    (REPORT_DIR / "phase1_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_map(stops_demand: gpd.GeoDataFrame, pois_jobs: gpd.GeoDataFrame) -> None:
    import folium

    bbox = stops_demand.total_bounds
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    colormap = folium.LinearColormap(
        ["#2c7bb6", "#ffffbf", "#d7191c"],
        vmin=0, vmax=max(float(stops_demand["population"].max()), 1e-6),
        caption="Население у остановки",
    )

    for _, row in stops_demand.iterrows():
        pt = row.geometry.centroid
        if pt.geom_type != "Point":
            continue
        folium.CircleMarker(
            location=[pt.y, pt.x],
            radius=4,
            color=colormap(row["population"]),
            fill=True,
            fillOpacity=0.5,
            popup=f"population={row['population']:.0f}, jobs={row['jobs']:.0f}",
        ).add_to(m)

    colormap.add_to(m)
    m.save(str(REPORT_DIR / "phase1_map.html"))
    print(f"Map saved: {REPORT_DIR / 'phase1_map.html'}")


if __name__ == "__main__":
    report = run_phase1()
    print(json.dumps(report, indent=2, ensure_ascii=False))