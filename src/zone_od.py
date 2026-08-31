"""Trip distribution on independent transport zones using the real road graph."""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import CACHE_DIR, LAYERS_DIR, PROJ_EPSG, REPORT_DIR
from src.tndp.network import ROAD_SPEED_KMH
from src.zones import build_transport_zones

DECAY_RADIUS_KM = 5.5


def build_road_graph(roads: gpd.GeoDataFrame) -> nx.Graph:
    """Build the same real-road graph used for TNDP candidate generation."""
    graph = nx.Graph()
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
                u, v = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
                length_km = float(np.hypot(a[0] - b[0], a[1] - b[1])) / 1000.0
                minutes = length_km / speed * 60.0
                if not graph.has_edge(u, v) or minutes < graph[u][v]["time"]:
                    graph.add_edge(u, v, time=minutes, length_km=length_km)
    return graph


def zone_network_costs(zones: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> np.ndarray:
    """Calculate zone-to-zone shortest travel time on the real road graph."""
    graph = build_road_graph(roads)
    nodes = list(graph.nodes)
    if not nodes:
        raise RuntimeError("Road graph is empty")
    tree = cKDTree(np.asarray(nodes, dtype=float))
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


def furness(T, origins, attractions, max_iter=300, tol=1e-5):
    """Balance the initial gravity matrix to zone productions/attractions."""
    T = np.asarray(T, dtype=np.float64).copy()
    for _ in range(max_iter):
        rows = T.sum(axis=1)
        T *= np.divide(origins, rows, out=np.zeros_like(origins), where=rows > 0)[:, None]
        cols = T.sum(axis=0)
        T *= np.divide(attractions, cols, out=np.zeros_like(attractions), where=cols > 0)[None, :]
        error = max(float(np.max(np.abs(T.sum(axis=1) - origins))),
                    float(np.max(np.abs(T.sum(axis=0) - attractions))))
        if error < tol:
            break
    return T


def build_zone_od(zones, cost, decay_radius_km=DECAY_RADIUS_KM):
    """Build a gravity OD matrix using network travel time as impedance."""
    production = zones.production.to_numpy(dtype=float)
    attraction = zones.attraction.to_numpy(dtype=float)
    if production.sum() <= 0 or attraction.sum() <= 0:
        raise RuntimeError("Zones contain no production or attraction")
    production *= attraction.sum() / production.sum()
    beta = 1.0 / (decay_radius_km / 25.0 * 60.0)
    impedance = np.where(np.isfinite(cost), cost, 1e6)
    kernel = np.exp(-beta * impedance)
    np.fill_diagonal(kernel, 0.0)
    kernel[~np.isfinite(cost)] = 0.0
    return furness(np.outer(production, attraction) * kernel, production, attraction)


def save_zone_od(zones, matrix, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = zones.zone_id.astype(int).to_numpy()
    pd.DataFrame(matrix, index=ids, columns=ids).to_parquet(out_dir / "od_matrix.parquet")
    rows, cols = np.nonzero(matrix > 0)
    pd.DataFrame({"orig_zone": ids[rows], "dest_zone": ids[cols], "trips": matrix[rows, cols]}) \
        .query("orig_zone != dest_zone") \
        .sort_values("trips", ascending=False) \
        .to_parquet(out_dir / "od_pairs.parquet", index=False)


def run_zone_od(zone_size_m=750.0, force=False):
    out = CACHE_DIR / "zone_od"
    report_path = out / "zone_od_report.json"
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    zones = build_transport_zones(size_m=zone_size_m, force=force)
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    t0 = time.time()
    cost = zone_network_costs(zones, roads)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "zone_travel_time_min.npy", cost)
    matrix = build_zone_od(zones, cost)
    save_zone_od(zones, matrix, out)
    total = max(float(matrix.sum()), 1.0)
    report = {
        "n_zones": int(len(zones)),
        "total_trips": float(matrix.sum()),
        "od_pairs": int((matrix > 0).sum() - len(matrix)),
        "avg_network_time_min": float((matrix * np.where(np.isfinite(cost), cost, 0)).sum() / total),
        "runtime_sec": round(time.time() - t0, 2),
        "zone_size_m": zone_size_m,
        "source": "transport zones + real road graph",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "zone_od_report.md").write_text("\n".join([
        "# OD-матрица транспортных зон", "",
        f"- Зон: **{report['n_zones']:,}**",
        f"- Поездок: **{report['total_trips']:,.0f}**",
        f"- OD-пар: **{report['od_pairs']:,}**",
        f"- Среднее время по сети: **{report['avg_network_time_min']:.1f} мин**",
        f"- Размер зоны: **{zone_size_m:.0f} м**",
    ]), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(run_zone_od())
