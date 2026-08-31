"""Trip distribution on independent transport zones.

The OD matrix is indexed by stable ``zone_id`` values. Transit stops are not
used as demand zones anymore. Costs are calculated on the real road graph by
snapping zone centroids to road vertices and running Dijkstra only from zone
origins.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree

from config import CACHE_DIR, LAYERS_DIR, PROJ_EPSG, REPORT_DIR
from src.phase2 import ROAD_SPEED_KMH
from src.zones import build_transport_zones

DECAY_RADIUS_KM = 5.5


def build_road_graph(roads: gpd.GeoDataFrame) -> nx.Graph:
    """Build a graph from every real road geometry vertex."""
    g = nx.Graph()
    projected = roads.to_crs(PROJ_EPSG).explode(index_parts=False, ignore_index=True)
    for _, row in projected.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        speed = float(ROAD_SPEED_KMH.get(str(row.get("highway") or "").lower(), 30.0))
        for line in lines:
            coords = list(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                if a[:2] == b[:2]:
                    continue
                x1, y1 = float(a[0]), float(a[1])
                x2, y2 = float(b[0]), float(b[1])
                length_km = float(np.hypot(x1 - x2, y1 - y2)) / 1000.0
                minutes = length_km / speed * 60.0
                if g.has_edge((x1, y1), (x2, y2)):
                    if minutes < g[(x1, y1)][(x2, y2)]["time"]:
                        g[(x1, y1)][(x2, y2)].update(time=minutes, length_km=length_km)
                else:
                    g.add_edge((x1, y1), (x2, y2), time=minutes, length_km=length_km)
    return g


def zone_network_costs(zones: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> np.ndarray:
    """Return zone-to-zone shortest travel time in minutes on the road graph."""
    graph = build_road_graph(roads)
    nodes = list(graph.nodes)
    if not nodes:
        raise RuntimeError("Road graph is empty")
    arr = np.asarray(nodes, dtype=float)
    tree = cKDTree(arr)
    centroids = zones.geometry.centroid
    xy = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    _, nearest = tree.query(xy, k=1)
    snapped = [nodes[int(i)] for i in nearest]

    n = len(zones)
    cost = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(cost, 0.0)
    for i, source in enumerate(snapped):
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="time")
        for j, target in enumerate(snapped):
            value = lengths.get(target)
            if value is not None:
                cost[i, j] = float(value)
    return cost


def furness(T: np.ndarray, origins: np.ndarray, attractions: np.ndarray,
            max_iter: int = 300, tol: float = 1e-5) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).copy()
    for _ in range(max_iter):
        rs = T.sum(axis=1)
        T *= np.divide(origins, rs, out=np.zeros_like(origins), where=rs > 0)[:, None]
        cs = T.sum(axis=0)
        T *= np.divide(attractions, cs, out=np.zeros_like(attractions), where=cs > 0)[None, :]
        err = max(
            float(np.max(np.abs(T.sum(axis=1) - origins), initial=0.0)),
            float(np.max(np.abs(T.sum(axis=0) - attractions), initial=0.0)),
        )
        if err < tol:
            break
    return T


def build_zone_od(zones: gpd.GeoDataFrame, cost: np.ndarray,
                  decay_radius_km: float = DECAY_RADIUS_KM) -> np.ndarray:
    """Gravity model with network travel time as impedance."""
    production = zones.production.to_numpy(dtype=float)
    attraction = zones.attraction.to_numpy(dtype=float)
    total_attr = attraction.sum()
    if total_attr <= 0 or production.sum() <= 0:
        raise RuntimeError("Zones contain no production/attraction")
    production *= total_attr / production.sum()
    beta = 1.0 / (decay_radius_km / 25.0 * 60.0)
    kernel = np.exp(-beta * np.where(np.isfinite(cost), cost, 1e6))
    np.fill_diagonal(kernel, 0.0)
    kernel[~np.isfinite(cost)] = 0.0
    T = np.outer(production, attraction) * kernel
    return furness(T, production, attraction)


def save_zone_od(zones: gpd.GeoDataFrame, T: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(T, index=zones.zone_id.astype(int), columns=zones.zone_id.astype(int))
    matrix.to_parquet(out_dir / "od_matrix.parquet")
    rows, cols = np.nonzero(T > 0)
    long = pd.DataFrame({
        "orig_zone": zones.zone_id.to_numpy()[rows].astype(int),
        "dest_zone": zones.zone_id.to_numpy()[cols].astype(int),
        "trips": T[rows, cols],
    })
    long = long[long.orig_zone != long.dest_zone].sort_values("trips", ascending=False)
    long.to_parquet(out_dir / "od_pairs.parquet", index=False)


def run_zone_od(zone_size_m: float = 750.0, force: bool = False) -> dict:
    out = CACHE_DIR / "zone_od"
    report_path = out / "zone_od_report.json"
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))

    zones = build_transport_zones(size_m=zone_size_m, force=force)
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    t0 = time.time()
    cost = zone_network_costs(zones, roads)
    np.save(out / "zone_travel_time_min.npy", cost)
    T = build_zone_od(zones, cost)
    save_zone_od(zones, T, out)
    report = {
        "n_zones": int(len(zones)),
        "total_production": float(T.sum(axis=1).sum()),
        "total_trips": float(T.sum()),
        "od_pairs": int((T > 0).sum() - len(T)),
        "avg_network_time_min": float((T * np.where(np.isfinite(cost), cost, 0)).sum() / max(T.sum(), 1.0)),
        "runtime_sec": round(time.time() - t0, 2),
        "zone_size_m": zone_size_m,
        "source": "real road graph + transport zones",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "zone_od_report.md").write_text("\n".join([
        "# OD-матрица транспортных зон",
        "",
        f"- Зон: **{report['n_zones']:,}**",
        f"- Поездок: **{report['total_trips']:,.0f}**",
        f"- OD-пар: **{report['od_pairs']:,}**",
        f"- Среднее время по сети: **{report['avg_network_time_min']:.1f} мин**",
        f"- Размер зоны: **{zone_size_m:.0f} м**",
    ]), encoding="utf-8")
    return report
