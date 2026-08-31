"""Phase 3 (real data): build routes from voronezh_routes_terminals.geojson.

Reads the repository-tracked reference stop/terminal file, orders stops within
 each route, filters out routes shorter than ROUTE_MIN_LENGTH_KM (2.5 km), and
produces the same output format as phase3 so that Phase 4 can consume it.
"""

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial as spatial
from shapely.geometry import LineString, Point

from config import CACHE_DIR, LAYERS_DIR, PROJ_EPSG, REFERENCE_ROUTES_PATH, REPORT_DIR

ROUTE_MIN_LENGTH_KM = 2.5


def _match_stops(ref_gdf, part_stops, max_snap_m=100.0):
    ref_proj = ref_gdf.to_crs(PROJ_EPSG)
    part_proj = part_stops.to_crs(PROJ_EPSG)
    ref_pts = np.column_stack([ref_proj.geometry.x, ref_proj.geometry.y])
    part_pts = np.column_stack([part_proj.geometry.x, part_proj.geometry.y])
    tree = spatial.cKDTree(part_pts)
    dist, idx = tree.query(ref_pts, k=1)
    stop_idx = np.where(dist <= max_snap_m, idx, np.nan)
    out = ref_gdf.copy()
    out["stop_idx"] = stop_idx
    out["snap_m"] = dist
    print(f"  Matched {(~np.isnan(stop_idx)).sum()}/{len(out)} reference stops (within {max_snap_m:.0f} m)")
    return out


def _order_nearest_neighbor(pts_2d, start, end):
    n = len(pts_2d)
    visited = {start}
    order = [start]
    current = start
    while len(visited) < n:
        best = None
        best_d = None
        for j in range(n):
            if j in visited:
                continue
            d = np.hypot(pts_2d[j][0] - pts_2d[current][0], pts_2d[j][1] - pts_2d[current][1])
            if best_d is None or d < best_d:
                best_d, best = d, j
        if best is None:
            break
        order.append(best)
        visited.add(best)
        current = best
    if end not in visited and end != start:
        order.append(end)
    return order


def _dedupe_names(stop_list, part_stops):
    seen = set()
    out = []
    for s in stop_list:
        name = part_stops.iloc[s].get("name")
        if name is None or pd.isna(name):
            out.append(s)
            continue
        key = str(name).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _build_routes(ref_gdf, part_stops):
    part_proj = part_stops.to_crs(PROJ_EPSG)
    part_pts = np.column_stack([part_proj.geometry.x, part_proj.geometry.y])
    route_ids = {str(r) for routes_list in ref_gdf["routes"] for r in routes_list}
    print(f"  Reference routes: {len(route_ids)}")

    all_routes = []
    route_info = {}
    for rid in sorted(route_ids, key=lambda x: (len(x), x)):
        route_ref = ref_gdf[ref_gdf["routes"].apply(lambda lst: rid in [str(x) for x in lst])]
        matched = route_ref.dropna(subset=["stop_idx"])
        if len(matched) < 2:
            continue
        matched_idx = matched["stop_idx"].astype(int).tolist()
        terminal_global = set(matched.loc[matched["is_terminal"] == True, "stop_idx"].astype(int).tolist())
        term_stop_ids = [s for s in matched_idx if s in terminal_global]

        candidates = term_stop_ids if len(term_stop_ids) >= 2 else matched_idx
        start, end, best_d = candidates[0], candidates[-1], 0.0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                d = np.hypot(part_pts[candidates[i]][0] - part_pts[candidates[j]][0],
                             part_pts[candidates[i]][1] - part_pts[candidates[j]][1])
                if d > best_d:
                    best_d, start, end = d, candidates[i], candidates[j]

        local = {s: i for i, s in enumerate(matched_idx)}
        ordered = [matched_idx[i] for i in _order_nearest_neighbor(part_pts[matched_idx], local[start], local[end])]
        ordered = _dedupe_names(ordered, part_stops)
        if len(ordered) < 2:
            continue
        line = LineString([(part_pts[s][0], part_pts[s][1]) for s in ordered])
        length_km = line.length / 1000.0
        if length_km < ROUTE_MIN_LENGTH_KM:
            continue
        all_routes.append((ordered, start, end))
        route_info[rid] = {"n_stops": len(ordered), "length_km_air": round(length_km, 2), "terminals": [start, end]}

    print(f"  Routes after filter (>={ROUTE_MIN_LENGTH_KM} km): {len(all_routes)}")
    return all_routes, route_info


def _build_network_struct(roads):
    from src.phase2 import build_road_graph
    g = build_road_graph(roads)
    node_list = list(g.nodes)
    node_arr = np.array(node_list, dtype=float)
    tree = spatial.cKDTree(node_arr)
    return g, node_arr, node_list, tree


def _network_order(stop_list, start, end, node_of, graph, node_list, stop_locs):
    if start not in node_of or end not in node_of:
        return stop_list, []
    try:
        path = nx.shortest_path(graph, node_list[node_of[start]], node_list[node_of[end]], weight="weight")
    except Exception:
        return stop_list, []
    if len(path) < 2:
        return stop_list, []
    path_pts = np.array(path, dtype=float)

    def idx_of(s):
        loc = stop_locs[s]
        d2 = ((path_pts - loc) ** 2).sum(axis=1)
        return int(np.argmin(d2)), float(np.sqrt(d2.min()))

    scored = [(s, idx_of(s)) for s in stop_list]
    scored.sort(key=lambda t: (t[1][0], t[1][1]))
    ordered = [s for s, _ in scored]
    if start in ordered and ordered[0] != start:
        ordered.remove(start)
        ordered.insert(0, start)
    if end in ordered and ordered[-1] != end:
        ordered.remove(end)
        ordered.append(end)
    return ordered, path


def _area_km2():
    try:
        from src.boundary import load_boundary
        from config import NAMES
        b = load_boundary(CACHE_DIR / f"{NAMES['boundary']}.geojson")
        return float(b.to_crs(PROJ_EPSG).area.iloc[0] / 1e6)
    except Exception:
        return 599.0


def run_phase3_real(force=False):
    out_dir = CACHE_DIR / "phase3_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"routes": out_dir / "routes.parquet", "flat": out_dir / "routes_flat.parquet",
             "report": out_dir / "phase3_report.json", "stops_pos": out_dir / "stops_pos.parquet"}
    if all(p.exists() for p in paths.values()) and not force:
        return json.load(open(paths["report"], encoding="utf-8"))

    ref_path = REFERENCE_ROUTES_PATH
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    ref_gdf = gpd.read_file(ref_path)
    part_stops = gpd.read_parquet(CACHE_DIR / "phase1_real" / "stops_demand.parquet")
    part_stops = part_stops[part_stops["radius_m"].notna()].reset_index(drop=True)
    print(f"Reference stops: {len(ref_gdf)}, participating stops: {len(part_stops)}")

    ref_gdf = _match_stops(ref_gdf, part_stops, max_snap_m=100.0)
    matched = ref_gdf.dropna(subset=["stop_idx"]).drop_duplicates(subset=["stop_idx"], keep="first")
    print(f"  Unique matched stops: {len(matched)}")
    all_routes, route_info = _build_routes(ref_gdf, part_stops)

    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    graph, node_arr, node_list, tree = _build_network_struct(roads)
    print(f"  Road graph nodes: {len(node_list)}")
    stop_proj = part_stops.geometry.to_crs(PROJ_EPSG)
    stop_pts = np.column_stack([stop_proj.geometry.x, stop_proj.geometry.y])

    route_feats, flat_rows = [], []
    for ri, (route, start, end) in enumerate(all_routes):
        node_of = {}
        for s in route:
            _, i = tree.query(stop_pts[s], k=1)
            node_of[s] = int(i)
        ordered, spine = _network_order(route, start, end, node_of, graph, node_list, stop_pts)
        line = LineString(spine) if len(spine) >= 2 else LineString([tuple(stop_pts[start]), tuple(stop_pts[end])])
        air_km = line.length / 1000.0
        route_feats.append({"route_id": ri, "n_stops": len(ordered), "length_km": air_km,
                            "length_km_nonlin": air_km * 2.0, "geometry": line})
        for order, stop_idx in enumerate(ordered):
            flat_rows.append({"route_id": ri, "order": order, "stop_idx": stop_idx,
                              "osm_id": part_stops.iloc[stop_idx]["osm_id"],
                              "name": part_stops.iloc[stop_idx].get("name")})

    route_gdf = gpd.GeoDataFrame(route_feats, geometry="geometry", crs=PROJ_EPSG).to_crs("EPSG:4326")
    flat = pd.DataFrame(flat_rows)
    route_gdf.to_parquet(paths["routes"])
    flat.to_parquet(paths["flat"])

    used_stops = flat["stop_idx"].unique()
    keep = [c for c in ("osm_id", "name", "kind", "is_terminal", "n_routes", "population", "jobs", "radius_m", "geometry") if c in part_stops.columns]
    part_stops.iloc[used_stops][keep].to_parquet(paths["stops_pos"])

    n_stops_file = int(len(ref_gdf))
    from src.phase3 import K1, K2, K3, calc_route_count
    p1 = json.load(open(CACHE_DIR / "phase1_real" / "phase1_report.json", encoding="utf-8"))
    pop_1000 = p1["population_sum_by_stop"] / 1000.0
    area_km2 = _area_km2()
    interchange = 13.0
    m = calc_route_count(pop_1000, area_km2, n_stops_file, interchange)
    n_routes_formula = int(round(m))

    report = {
        "n_ref_stops": len(ref_gdf), "n_ref_routes": len(route_info), "n_matched_stops": len(matched),
        "n_routes": len(all_routes), "n_routes_formula": n_routes_formula, "route_count_formula_m": round(m, 2),
        "formula": {"N_thousands": round(pop_1000, 1), "S_km2": area_km2, "O_stops_file": n_stops_file,
                    "interchange": interchange, "k1": K1, "k2": K2, "k3": K3},
        "n_stops_served": len(used_stops),
        "avg_route_km_air": round(float(route_gdf["length_km"].mean()), 2) if len(route_gdf) else 0,
        "total_route_km_air": round(float(route_gdf["length_km"].sum()), 1),
        "source": str(ref_path.relative_to(REFERENCE_ROUTES_PATH.parent)),
    }
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    (REPORT_DIR / "phase3_real_report.md").write_text("\n".join([
        "# Фаза 3 (реальные маршруты)", "", f"- Источник: **{report['source']}**",
        f"- Остановок в файле: {report['n_ref_stops']}", f"- Маршрутов в файле: {report['n_ref_routes']}",
        f"- Совпавших остановок: {report['n_matched_stops']}", f"- Построено маршрутов: {report['n_routes']}",
        f"- Остановок охвачено: {report['n_stops_served']}", f"- Ср. длина: {report['avg_route_km_air']} км",
        f"- Суммарная длина: {report['total_route_km_air']} км", "",
    ]), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_phase3_real(force=True), indent=2, ensure_ascii=False))
