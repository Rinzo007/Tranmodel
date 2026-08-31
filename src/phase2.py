"""Phase 2: gravity-based trip distribution matrix T_ij = O_i * D_j * f(d_ij).

Inputs:
  - phase1/stops_demand.parquet   (participating stops with population & jobs)
  - layers/roads.parquet          (road network for network cost matrix)

Costs:
  - by air (straight-line) distance;
  - by real road network (shortest travel time).

Gravity model:
  T_ij = O_i * D_j * exp(-beta * t_ij)
Balanced to origin (O) and destination (D) totals with the Furness method.

Outputs (data/cache/phase2/):
  - stops_matrix.parquet          participating stops + indices
  - od_dist_air.npz / od_dist_net.npz   distance/time matrices
  - matrix_od.parquet             sparse long-form OD pairs with trips
  - phase2_report.json / .md
"""

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
import scipy.spatial as spatial
from shapely.geometry import Point, LineString

from config import CACHE_DIR, LAYERS_DIR, PROJ_EPSG, REPORT_DIR


GRAVITY_DECAY_RADIUS_KM = 5.5   # distance at which flow decays to 1/e
DEFAULT_AVG_TRIP_TIME = 30.0     # target avg trip time, min (log only)


class Phase2Error(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Road network graph + travel-time cost matrix
# ---------------------------------------------------------------------------
# average speed by road class (km/h)
ROAD_SPEED_KMH = {
    "motorway": 90, "motorway_link": 50,
    "trunk": 70, "trunk_link": 40,
    "primary": 60, "primary_link": 35,
    "secondary": 50, "secondary_link": 30,
    "tertiary": 40, "tertiary_link": 25,
    "unclassified": 30, "residential": 30, "living_street": 20,
    "service": 20, "road": 30, "track": 15, "pedestrian": 5,
    "footway": 5, "cycleway": 15, "services": 20,
}

TRAVEL_SPEED_WALK_KMH = 4.5  # access/egress to nearest road node


def build_road_graph(roads: gpd.GeoDataFrame) -> nx.Graph:
    """Build an undirected graph from road centre-lines, edge weight = minutes."""
    g = nx.Graph()
    g2_roads = roads.to_crs(PROJ_EPSG)
    for _, row in g2_roads.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type not in ("LineString", "MultiLineString"):
            continue
        speed = ROAD_SPEED_KMH.get((row.get("highway") or "").lower(), 30)
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for line in lines:
            if line.length <= 0:
                continue
            coords = list(line.coords)
            for (a, b) in zip(coords[:-1], coords[1:]):
                dist_m = ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
                minutes = (dist_m / 1000.0) / speed * 60.0
                if a == b:
                    continue
                if g.has_edge(a, b):
                    g[a][b]["weight"] = min(g[a][b]["weight"], minutes)
                else:
                    g.add_edge(a, b, weight=minutes)
    return g


def road_cost_matrix(
    stops: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    use_network: bool = True,
) -> tuple[np.ndarray, list]:
    """Return (cost_matrix[origin x destination] minutes, node_labels).

    If use_network, computes shortest-path time over the road graph from road
    nodes nearest each stop (walking access to/from the road network is ignored
    here and captured separately if needed). Otherwise, straight-line commute.
    """
    stop_proj = stops.geometry.to_crs(PROJ_EPSG)
    stop_pts = np.column_stack([stop_proj.geometry.x, stop_proj.geometry.y])

    n = len(stops)
    if use_network:
        g = build_road_graph(roads)
        nodes = list(g.nodes)
        if not nodes:
            raise Phase2Error("Empty road graph — cannot build network cost matrix")
        node_arr = np.array(nodes, dtype=float)

        # snap each stop to nearest road node
        tree = spatial.cKDTree(node_arr)
        snap_dist, snap_idx = tree.query(stop_pts, k=1)
        snap_nodes = [nodes[i] for i in snap_idx]

        # all-pairs shortest path (unweighted Dijkstra per source)
        length = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
        matrix = np.full((n, n), np.inf)
        for i in range(n):
            row = length.get(snap_nodes[i], {})
            for j in range(n):
                tj = row.get(snap_nodes[j])
                if tj is not None:
                    matrix[i, j] = tj
        # diagonal -> 0
        np.fill_diagonal(matrix, 0.0)
    else:
        # straight-line distance -> time at walk? Use road-free: air distance in km
        # converted to minutes at an assumed speed (e.g. transit=25 km/h straight)
        air_km = spatial.distance.cdist(
            stop_pts / 1000.0, stop_pts / 1000.0, metric="euclidean"
        )
        matrix = air_km / (25.0 / 60.0)  # minutes at 25 km/h straight
        np.fill_diagonal(matrix, 0.0)
        snap_dist = np.zeros(n)

    return matrix, stop_proj.index.tolist()


# ---------------------------------------------------------------------------
# Gravity + Furness balancing
# ---------------------------------------------------------------------------
def gravity_matrix(
    O: np.ndarray,
    D: np.ndarray,
    dist_km: np.ndarray,
    decay_radius_km: float,
) -> np.ndarray:
    """Unbalanced gravity by DISTANCE decay:
    T_ij = O_i * D_j * exp(-d_ij / R), R = decay radius (km).
    """
    kernel = np.exp(-dist_km / decay_radius_km)
    np.fill_diagonal(kernel, 0.0)
    return np.outer(O, D) * kernel


def furness_balance(T: np.ndarray, O: np.ndarray, D: np.ndarray,
                    max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """Iteratively balance T to row totals O and column totals D."""
    T = T.copy()
    for _ in range(max_iter):
        row_sum = T.sum(axis=1)
        row_f = np.divide(O, row_sum, out=np.zeros_like(O), where=row_sum > 0)
        T = T * row_f[:, None]
        col_sum = T.sum(axis=0)
        col_f = np.divide(D, col_sum, out=np.zeros_like(D), where=col_sum > 0)
        T = T * col_f[None, :]
        rerr = float(np.max(np.abs(T.sum(axis=1) - O))) if T.shape[0] > 0 else 0.0
        cerr = float(np.max(np.abs(T.sum(axis=0) - D))) if T.shape[1] > 0 else 0.0
        if max(rerr, cerr) < tol:
            break
    return T


def average_trip_time(T: np.ndarray, cost: np.ndarray) -> float:
    tot = T.sum()
    if tot <= 0:
        return 0.0
    return float((T * cost).sum() / tot)


# ---------------------------------------------------------------------------
# Long-form matrix output
# ---------------------------------------------------------------------------
def to_long_form(T: np.ndarray, stops_idx: list) -> pd.DataFrame:
    """Return sparse long-form DataFrame of OD trips (drop zero / self pairs)."""
    n = len(T)
    i_idx, j_idx = np.nonzero(T)
    rows = []
    for i, j in zip(i_idx, j_idx):
        if i == j:
            continue
        rows.append({"orig": stops_idx[i], "dest": stops_idx[j], "trips": float(T[i, j])})
    df = pd.DataFrame(rows)
    df = df[df["trips"] > 0].reset_index(drop=True)
    df = df.sort_values("trips", ascending=False).reset_index(drop=True)
    return df


def run_phase2(
    decay_radius_km: float = GRAVITY_DECAY_RADIUS_KM,
    use_network: bool = False,
    force: bool = False,
) -> dict:
    out_dir = CACHE_DIR / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stops": out_dir / "stops_matrix.parquet",
        "od": out_dir / "matrix_od.parquet",
        "air": out_dir / "od_costs_air.npy",
        "net": out_dir / "od_costs_net.npy",
        "report": out_dir / "phase2_report.json",
    }
    if all(p.exists() for p in paths.values()) and not force:
        return json.load(open(paths["report"], encoding="utf-8"))

    stops = gpd.read_parquet(CACHE_DIR / "phase1_real" / "stops_demand.parquet")
    stops = stops[stops["radius_m"].notna()].reset_index(drop=True)
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")

    n = len(stops)
    if n == 0:
        raise Phase2Error("No stops with radius — run Phase 1 first")
    stops_idx = list(stops.index)
    O = stops["population"].to_numpy(dtype=float)
    D = stops["jobs"].to_numpy(dtype=float)

    print(f"Participating stops: {n}; origins sum={O.sum():,.0f}, dests sum={D.sum():,.0f}")

    # Balance trip production origins to equal attraction destinations total
    # (Furness needs a common grand total). Scale origins by jobs/population.
    O_raw = O.copy()
    grand_total = float(D.sum())
    if O.sum() > 0:
        O = O * (grand_total / O.sum())
    print(f"Origins scaled to {O.sum():,.0f} (grand total = {grand_total:,.0f})")

    # Add straight-line distance (km) matrix for the DISTANCE-decay model.
    stop_proj = stops.geometry.to_crs(PROJ_EPSG)
    stop_xyz = np.column_stack([stop_proj.geometry.x / 1000.0,
                                stop_proj.geometry.y / 1000.0])
    dist_km = spatial.distance.cdist(stop_xyz, stop_xyz, metric="euclidean")
    np.fill_diagonal(dist_km, 0.0)
    np.save(paths["air"], dist_km)
    print(f"  straight-line distance matrix built ({dist_km.shape[0]}x{dist_km.shape[1]})")

    if use_network:
        print("Building road-network cost matrix (Dijkstra all-pairs) ...")
        t0 = time.time()
        cost_net, _ = road_cost_matrix(stops, roads, use_network=True)
        np.save(paths["net"], cost_net)
        print(f"  network matrix {time.time()-t0:.0f}s")
    else:
        cost_net = None

    # Distance-decay gravity with fixed decay radius (km)
    decay_radius = decay_radius_km
    print(f"Computing gravity matrix (decay radius = {decay_radius} km) + Furness ...")
    T = gravity_matrix(O, D, dist_km, decay_radius)
    T = furness_balance(T, O, D)

    np.fill_diagonal(T, 0.0)
    od = to_long_form(T, stops_idx)
    od.to_parquet(paths["od"])
    print(f"  OD pairs: {len(od)}, total trips: {od['trips'].sum():,.0f}")

    keep = [c for c in ("osm_id", "name", "kind", "is_terminal", "n_routes",
                        "population", "jobs", "radius_m", "geometry")
            if c in stops.columns]
    stops[keep].to_parquet(paths["stops"])

    avg_dist_km = average_trip_time(T, dist_km)
    avg_net_min = average_trip_time(T, cost_net) if (cost_net is not None and np.isfinite(cost_net).all()) else None

    report = {
        "n_stops": n,
        "total_population": round(float(O_raw.sum()), 1),
        "total_jobs": round(float(D.sum()), 1),
        "grand_total_trips": round(grand_total, 1),
        "decay_radius_km": decay_radius,
        "avg_dist_km": round(float(avg_dist_km), 2),
        "avg_net_min": round(float(avg_net_min), 2) if avg_net_min is not None else None,
        "total_trips": round(float(T.sum()), 1),
        "n_od_pairs": int(len(od)),
        "max_od_trips": round(float(od["trips"].max()), 1) if len(od) else 0.0,
        "cost_mode": "network" if use_network else "air",
    }
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_markdown(report)
    _write_map(stops, od)
    return report


def _write_markdown(report: dict) -> None:
    lines = [
        "# Фаза 2 — Матрица корреспонденций (гравитационная модель)",
        "",
        f"- Участвующих остановок: **{report['n_stops']}**",
        f"- Население (генерация): {report['total_population']:,.0f}",
        f"- Рабочие места (притяжение): {report['total_jobs']:,.0f}",
        f"- Радиус затухания: **{report['decay_radius_km']} км**",
        f"- Средняя дальность поездки (по воздушной линии): {report['avg_dist_km']} км",
        f"- Итоговый объём поездок (сумма по матрице): **{report['total_trips']:,.0f}**",
        f"- Пара OD с поездками: {report['n_od_pairs']:,}",
        f"- Максимум по одной паре: {report['max_od_trips']:,.0f}",
        "",
    ]
    (REPORT_DIR / "phase2_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_map(stops: gpd.GeoDataFrame, od: pd.DataFrame) -> None:
    import folium

    bbox = stops.total_bounds
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    # Top OD desire lines
    top = od.head(1500)
    if len(top):
        stop_pos = {i: (stops.loc[i].geometry.y, stops.loc[i].geometry.x) for i in stops.index}
        colormap = folium.LinearColormap(["#ffffcc", "#d8b365", "#a6611a", "#543005"],
                                         vmin=0, vmax=max(top["trips"].max(), 1), caption="Poezdki")
        for _, row in top.iterrows():
            a = stop_pos.get(row["orig"])
            b = stop_pos.get(row["dest"])
            if not a or not b:
                continue
            folium.PolyLine([a, b], color=colormap(row["trips"]), weight=1.2, opacity=0.5).add_to(m)
        colormap.add_to(m)

    for _, row in stops.iterrows():
        p = row.geometry.centroid
        folium.CircleMarker(location=[p.y, p.x], radius=3, color="#000", fill=True,
                            fillOpacity=0.4, popup=f"pop={row['population']:.0f} jobs={row['jobs']:.0f}").add_to(m)

    m.save(str(REPORT_DIR / "phase2_map.html"))
    print(f"Map saved: {REPORT_DIR / 'phase2_map.html'}")


if __name__ == "__main__":
    report = run_phase2()
    print(json.dumps(report, indent=2, ensure_ascii=False))