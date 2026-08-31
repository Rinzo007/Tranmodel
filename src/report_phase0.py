"""Build a report + map for Phase 0 (data preparation).

Demonstrates that all Phase 0 layers are ready and writes:
  - a summary JSON + Markdown report in data/report/
  - an interactive HTML map in data/report/phase0_map.html
"""

import json

import folium
import geopandas as gpd
import pandas as pd

from src.boundary import load_boundary
from src.osm_extract import extract_layers
from src.population import compute_population, load_population_tif

from config import (
    CACHE_DIR,
    LAYERS_DIR,
    NAMES,
    PBF_PATH,
    PROJ_EPSG,
    REPORT_DIR,
    ensure_dirs,
)

import numpy as np


def roads_stats(roads: gpd.GeoDataFrame) -> dict:
    roads = roads.copy()
    roads["km"] = roads.geometry.to_crs(PROJ_EPSG).length / 1000.0
    total = float(roads["km"].sum())
    by_class = roads.groupby("highway")["km"].sum().sort_values(ascending=False)
    return {
        "count": int(len(roads)),
        "total_km": round(total, 1),
        "named": int(roads["name"].notna().sum()),
        "oneway_share": round(float(roads["oneway"].eq("yes").mean()), 3),
        "by_class": {k: round(v, 1) for k, v in by_class.items()},
    }


def stops_stats(stops: gpd.GeoDataFrame) -> dict:
    counts = stops["kind"].value_counts().to_dict()
    modes = {}
    if counts.get("bus"):
        modes["bus"] = int(counts["bus"])
    if counts.get("rail"):
        modes["rail"] = int(counts["rail"])
    platforms = int(counts.get("platform", 0))
    return {
        "count": int(len(stops)),
        "modes": modes,
        "platforms": platforms,
        "named": int(stops["name"].notna().sum()),
    }


def pois_stats(pois: gpd.GeoDataFrame) -> dict:
    total = int(len(pois))
    by_key = pois["main_key"].value_counts().to_dict()
    top_values = (
        pois.groupby(["main_key", "value"]).size()
        .sort_values(ascending=False)
        .head(15)
        .to_dict()
    )
    top_values = {f"{k[0]}:{k[1]}": int(v) for k, v in top_values.items()}
    return {"count": total, "by_key": by_key, "top": top_values}


def population_raster_summary() -> dict:
    arr = load_population_tif()
    arr = arr[arr > 0]
    return {
        "n_cells": int(len(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def build_report(force: bool = False) -> dict:
    ensure_dirs()
    boundary_gdf = load_boundary(CACHE_DIR / f"{NAMES['boundary']}.geojson")
    boundary = boundary_gdf.geometry.iloc[0]

    layers = extract_layers(PBF_PATH, boundary, LAYERS_DIR, force=force)
    pop = compute_population(boundary_gdf)

    area_km2 = boundary_gdf.to_crs(PROJ_EPSG).area.iloc[0] / 1e6
    jobs_total = pop["total_population"] / 2.0

    report = {
        "city": boundary_gdf.display_name.iloc[0],
        "area_km2": round(area_km2, 1),
        "boundary": {"osm_id": int(boundary_gdf.osm_id.iloc[0]), "geom": boundary_gdf.geometry.iloc[0].geom_type},
        "population": {
            "total": pop["total_population"],
            "raster_summary": population_raster_summary(),
        },
        "jobs": {
            "total_formula_pop_div2": round(jobs_total, 1),
            "poi_count": int(len(layers["pois"])),
            "jobs_per_poi": round(jobs_total / len(layers["pois"]), 2) if len(layers["pois"]) else None,
        },
        "roads": roads_stats(layers["roads"]),
        "rail": {
            "count": int(len(layers["rail"])),
            "km": round(float(layers["rail"].to_crs(PROJ_EPSG).length.sum() / 1000.0), 1),
        },
        "waterways": {
            "count": int(len(layers["waterways"])),
            "km": round(float(layers["waterways"].to_crs(PROJ_EPSG).length.sum() / 1000.0), 1),
        },
        "stops": stops_stats(layers["stops"]),
        "pois": pois_stats(layers["pois"]),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "phase0_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _write_markdown(report)
    _write_map(report, boundary_gdf, layers)
    return report


def _write_markdown(report: dict) -> None:
    p = report["population"]
    r = report["roads"]
    lines = [
        "# Фаза 0 — Подготовка данных (отчёт)",
        "",
        f"**Город:** {report['city']}",
        f"**Площадь:** {report['area_km2']} км²  ",
        f"**Граница:** OSM relation {report['boundary']['osm_id']} ({report['boundary']['geom']})",
        "",
        "## Население (WorldPop 100m)",
        f"- Всего жителей: **{p['total']:,.0f}**",
        f"- Ячеек с населением: {p['raster_summary']['n_cells']:,}",
        f"- Макс./среднее в ячейке: {p['raster_summary']['max']:.1f} / {p['raster_summary']['mean']:.1f}",
        "",
        "## Рабочие места",
        f"- Формула население/2: **{report['jobs']['total_formula_pop_div2']:,.0f}**",
        f"- POI: {report['jobs']['poi_count']}",
        f"- Раб. мест на POI: {report['jobs']['jobs_per_poi'] or 0:.2f}",
        "",
        "## Дорожная сеть",
        f"- Улиц: **{r['count']:,}** (всего {r['total_km']:.0f} км)",
        f"- С названием: {r['named']:,}",
        f"- Односторонних: {r['oneway_share']*100:.1f}%",
        "Классы:", "",
    ]
    for k, km in list(r["by_class"].items())[:12]:
        lines.append(f"  - `{k}`: {km:.1f} км")
    lines += [
        "",
        "## Ж/д и водные пути",
        f"- Ж/д: {report['rail']['count']} сегментов, {report['rail']['km']:.0f} км",
        f"- Водные: {report['waterways']['count']} сегментов, {report['waterways']['km']:.0f} км",
        "",
        "## Остановки",
        f"- Всего: **{report['stops']['count']}**",
        f"- Автобус: {report['stops']['modes'].get('bus', 0)}, Ж/д и РТ: {report['stops']['modes'].get('rail', 0)}, платформы: {report['stops']['platforms']}",
        f"- С названием: {report['stops']['named']}",
        "",
        "## POI",
        f"- Всего: **{report['pois']['count']}**",
        "По категориям:",
        "",
    ]
    for k, v in report["pois"]["by_key"].items():
        lines.append(f"  - `{k}`: {v}")
    lines += [
        "",
        "## Топ-15 POI",
        "",
    ]
    for k, v in report["pois"]["top"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    (REPORT_DIR / "phase0_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_map(report: dict, boundary_gdf: gpd.GeoDataFrame, layers: dict) -> None:
    b = boundary_gdf.geometry.iloc[0].bounds
    center = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    _add_geometry(m, boundary_gdf, "Граница Воронежа",
                  lambda _: {"color": "#d62728", "weight": 2, "fillOpacity": 0.05})

    _add_lines(m, layers["roads"], "Дороги", "#7f7f7f", 1.4)
    _add_lines(m, layers["rail"], "Ж/д", "#2ca02c", 2)
    _add_lines(m, layers["waterways"], "Водные пути", "#1f77b4", 2.5)

    stops = layers["stops"]
    pop_cols = [c for c in ["name", "kind"] if c in stops.columns]
    if pop_cols:
        _points = stops[pop_cols + ["geometry"]].head(3000)
        folium.GeoJson(
            _points.__geo_interface__,
            name="Остановки",
            marker=folium.CircleMarker(radius=4, color="black", fill=True, fillOpacity=1, fillColor="#ff7f0e"),
            popup=folium.GeoJsonPopup(fields=pop_cols),
        ).add_to(m)

    poi_cols = [c for c in ["name", "main_key", "value"] if c in layers["pois"].columns]
    if poi_cols:
        _pois = layers["pois"][poi_cols + ["geometry"]].head(5000)
        folium.GeoJson(
            _pois.__geo_interface__,
            name="POI",
            marker=folium.CircleMarker(radius=4, color="#9467bd", fill=True, fillOpacity=0.8, fillColor="#9467bd"),
            popup=folium.GeoJsonPopup(fields=poi_cols),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    out = REPORT_DIR / "phase0_map.html"
    m.save(str(out))
    report["map_path"] = str(out)
    print(f"Map saved: {out}")


def _add_geometry(m, gdf, name, style_fn):
    folium.GeoJson(
        gdf.geometry.__geo_interface__,
        name=name,
        style_function=style_fn,
    ).add_to(m)


def _add_lines(m, gdf, name, color, weight):
    if gdf.empty:
        return
    keep = [c for c in ["name", "osm_id"] if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]]
    folium.GeoJson(
        gdf.__geo_interface__,
        name=name,
        style_function=lambda _, c=color, w=weight: {"color": c, "weight": w, "opacity": 0.75},
    ).add_to(m)


if __name__ == "__main__":
    import pickle
    import json as _json

    layers = extract_layers(PBF_PATH, load_boundary(CACHE_DIR / f"{NAMES['boundary']}.geojson").geometry.iloc[0], LAYERS_DIR)
    for n, g in layers.items():
        with open(LAYERS_DIR / f"{n}.pkl", "wb") as f:
            pickle.dump(g, f)
    report = build_report()
    print(_json.dumps(report, indent=2, ensure_ascii=False))