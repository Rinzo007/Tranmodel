"""Phase 3 (real data): build routes from voronezh_routes_terminals.geojson.

Reads the reference stop/terminal file, orders stops within each route using
nearest-neighbor (terminal -> terminals), filters out routes shorter than
ROUTE_MIN_LENGTH_KM (2.5 km), and produces the same output format as phase3
so that Phase 4 can consume it.

Outputs (data/cache/phase3_real/):
  - routes.parquet        GeoDataFrame of routes (geometry=LineString)
  - routes_flat.parquet   long-form: route_id, order, stop_idx, osm_id, name
  - stops_pos.parquet     participating stops used by the routes
  - phase3_report.json    summary stats
"""

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial as spatial
from shapely.geometry import LineString, Point

from config import CACHE_DIR, LAYERS_DIR, PROJ_EPSG, REPORT_DIR

ROUTE_MIN_LENGTH_KM = 2.5


def _match_stops(
    ref_gdf: gpd.GeoDataFrame,
    part_stops: gpd.GeoDataFrame,
    max_snap_m: float = 100.0,
) -> gpd.GeoDataFrame:
    """Snap reference stops to participating stops by nearest distance.

    Returns a copy of ref_gdf with an added 'stop_idx' column pointing to
    the index in part_stops (or NaN if no match within max_snap_m).
    """
    ref_proj = ref_gdf.to_crs(PROJ_EPSG)
    part_proj = part_stops.to_crs(PROJ_EPSG)
    ref_pts = np.column_stack([ref_proj.geometry.x, ref_proj.geometry.y])
    part_pts = np.column_stack([part_proj.geometry.x, part_proj.geometry.y])
    tree = spatial.cKDTree(part_pts)
    dist, idx = tree.query(ref_pts, k=1)
    stop_idx = np.where(dist <= max_snap_m, idx, np.nan)
    ref_gdf = ref_gdf.copy()
    ref_gdf["stop_idx"] = stop_idx
    ref_gdf["snap_m"] = dist
    n_matched = int((~np.isnan(stop_idx)).sum())
    n_total = len(ref_gdf)
    print(f"  Matched {n_matched}/{n_total} reference stops "
          f"(within {max_snap_m:.0f} m)")
    return ref_gdf


def _order_nearest_neighbor(pts_2d: np.ndarray, start: int, end: int) -> list[int]:
    """Order all points via nearest-neighbor from start to end.

    Returns list of indices in visit order. start is first, end is last.
    """
    n = len(pts_2d)
    visited = set()
    order = [start]
    visited.add(start)
    current = start
    while len(visited) < n:
        # find nearest unvisited
        best = None
        best_d = None
        for j in range(n):
            if j in visited:
                continue
            d = np.hypot(pts_2d[j][0] - pts_2d[current][0],
                         pts_2d[j][1] - pts_2d[current][1])
            if best_d is None or d < best_d:
                best_d = d
                best = j
        if best is None:
            break
        order.append(best)
        visited.add(best)
        current = best
    if end not in visited and end != start:
        order.append(end)
    return order


def _dedupe_names(stop_list: list[int], part_stops: gpd.GeoDataFrame) -> list[int]:
    """Drop stops whose name already appeared earlier in the ordered route.

    Keeps the first occurrence of each stop name (case-insensitive) and removes
    later stops that would duplicate that name within the same route.
    """
    seen = set()
    out: list[int] = []
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


def _build_routes(
    ref_gdf: gpd.GeoDataFrame,
    part_stops: gpd.GeoDataFrame,
) -> tuple[list[list[int]], dict]:
    """Build ordered stop lists per route from the reference data.

    Returns (routes, info) where routes is a list of lists of stop_idx in
    part_stops, and info maps route_id to metadata.
    """
    part_proj = part_stops.to_crs(PROJ_EPSG)
    part_pts = np.column_stack([part_proj.geometry.x, part_proj.geometry.y])

    route_ids = set()
    for routes_list in ref_gdf["routes"]:
        for r in routes_list:
            route_ids.add(str(r))

    print(f"  Reference routes: {len(route_ids)}")

    all_routes = []
    route_info = {}
    for rid in sorted(route_ids, key=lambda x: (len(x), x)):
        # stops belonging to this route
        mask = ref_gdf["routes"].apply(lambda lst: rid in [str(x) for x in lst])
        route_ref = ref_gdf[mask].copy()
        matched = route_ref.dropna(subset=["stop_idx"])
        if len(matched) < 2:
            continue
        matched_idx = matched["stop_idx"].astype(int).tolist()

        # identify terminals (is_terminal=True) among matched stops
        terminal_global = set(
            matched.loc[matched["is_terminal"] == True, "stop_idx"].astype(int).tolist()
        )
        term_stop_ids = [s for s in matched_idx if s in terminal_global]

        if len(term_stop_ids) >= 2:
            best_d = 0
            start = term_stop_ids[0]
            end = term_stop_ids[-1]
            for i in range(len(term_stop_ids)):
                for j in range(i + 1, len(term_stop_ids)):
                    d = np.hypot(
                        part_pts[term_stop_ids[i]][0] - part_pts[term_stop_ids[j]][0],
                        part_pts[term_stop_ids[i]][1] - part_pts[term_stop_ids[j]][1])
                    if d > best_d:
                        best_d = d
                        start = term_stop_ids[i]
                        end = term_stop_ids[j]
        else:
            best_d = 0
            start = matched_idx[0]
            end = matched_idx[-1]
            for i in range(len(matched_idx)):
                for j in range(i + 1, len(matched_idx)):
                    d = np.hypot(
                        part_pts[matched_idx[i]][0] - part_pts[matched_idx[j]][0],
                        part_pts[matched_idx[i]][1] - part_pts[matched_idx[j]][1])
                    if d > best_d:
                        best_d = d
                        start = matched_idx[i]
                        end = matched_idx[j]

        # order all matched stops: nearest-neighbor from start
        # build local index mapping: global stop_idx -> local index
        local = {s: i for i, s in enumerate(matched_idx)}
        pts_local = part_pts[matched_idx]
        local_start = local[start]
        local_end = local[end]
        local_order = _order_nearest_neighbor(pts_local, local_start, local_end)
        ordered_global = [matched_idx[i] for i in local_order]

        # drop stops whose name repeats earlier in the route (keep first one)
        ordered_global = _dedupe_names(ordered_global, part_stops)
        if len(ordered_global) < 2:
            continue

        # compute route length (air distance)
        coords = [(part_pts[s][0], part_pts[s][1]) for s in ordered_global]
        line = LineString(coords)
        length_km = line.length / 1000.0

        if length_km < ROUTE_MIN_LENGTH_KM:
            continue

        all_routes.append((ordered_global, start, end))
        route_info[rid] = {
            "n_stops": len(ordered_global),
            "length_km_air": round(length_km, 2),
            "terminals": [start, end],
        }

    print(f"  Routes after filter (>={ROUTE_MIN_LENGTH_KM} km): {len(all_routes)}")
    return all_routes, route_info


def _build_network_struct(roads: gpd.GeoDataFrame):
    """Build road graph + KDTree for snapping stops to graph nodes.

    Returns (graph, node_coords, node_list, tree).
    """
    from src.phase2 import build_road_graph
    g = build_road_graph(roads)
    node_list = list(g.nodes)
    node_arr = np.array(node_list, dtype=float)
    tree = spatial.cKDTree(node_arr)
    return g, node_arr, node_list, tree


def _snap_to_nodes(part_stops: gpd.GeoDataFrame, stop_list, tree):
    """Snap each stop to nearest graph node index (in node_list order)."""
    proj = part_stops.geometry.to_crs(PROJ_EPSG)
    pts = np.column_stack([proj.geometry.x, proj.geometry.y])
    pts = pts[list(stop_list)]
    d, idx = tree.query(pts, k=1)
    return idx.tolist(), d.tolist()


def _network_order(stop_list, start, end, node_of, graph, node_list, stop_locs):
    """Order stops along the single shortest path start-terminal -> end-terminal.

    The route spine is one continuous road path P from A to B. Every stop is
    projected onto P (its position along the polyline), and stops are returned
    sorted by that position. Returns (ordered_stops, spine_path).
    """
    if start not in node_of or end not in node_of:
        return stop_list, []
    a = node_list[node_of[start]]
    b = node_list[node_of[end]]
    try:
        path = nx.shortest_path(graph, a, b, weight="weight")
    except Exception:
        return stop_list, []
    if len(path) < 2:
        return stop_list, []
    path_pts = np.array(path, dtype=float)

    # snap each stop to the nearest point on the spine; use its index as the order
    def idx_of(s):
        loc = stop_locs[s]
        d2 = ((path_pts - loc) ** 2).sum(axis=1)
        return int(np.argmin(d2)), float(np.sqrt(d2.min()))

    scored = [(s, idx_of(s)) for s in stop_list]
    scored.sort(key=lambda t: (t[1][0], t[1][1]))

    ordered = [s for s, _ in scored]
    # start first, end last (both on the spine at extremes)
    if start in ordered and ordered[0] != start:
        ordered.remove(start)
        ordered.insert(0, start)
    if end in ordered and ordered[-1] != end:
        ordered.remove(end)
        ordered.append(end)
    return ordered, path


def _area_km2() -> float:
    """Urban area (km^2) from the stored boundary in local UTM (EPSG:32637)."""
    try:
        from src.boundary import load_boundary
        from config import NAMES
        b = load_boundary(CACHE_DIR / f"{NAMES['boundary']}.geojson")
        return float(b.to_crs(PROJ_EPSG).area.iloc[0] / 1e6)
    except Exception:
        return 599.0


def run_phase3_real(force: bool = False) -> dict:
    out_dir = CACHE_DIR / "phase3_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "routes": out_dir / "routes.parquet",
        "flat": out_dir / "routes_flat.parquet",
        "report": out_dir / "phase3_report.json",
        "stops_pos": out_dir / "stops_pos.parquet",
    }
    if all(p.exists() for p in paths.values()) and not force:
        return json.load(open(paths["report"], encoding="utf-8"))

    ref_path = Path(r"D:\Programs\Project\voronezh_routes_terminals.geojson")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    ref_gdf = gpd.read_file(ref_path)
    part_stops = gpd.read_parquet(CACHE_DIR / "phase1_real" / "stops_demand.parquet")
    part_stops = part_stops[part_stops["radius_m"].notna()].reset_index(drop=True)

    print(f"Reference stops: {len(ref_gdf)}, participating stops: {len(part_stops)}")

    ref_gdf = _match_stops(ref_gdf, part_stops, max_snap_m=100.0)

    # only keep matched stops
    matched = ref_gdf.dropna(subset=["stop_idx"])
    # drop duplicates: same stop can appear in multiple routes; keep first
    matched = matched.drop_duplicates(subset=["stop_idx"], keep="first")
    print(f"  Unique matched stops: {len(matched)}")

    all_routes, route_info = _build_routes(ref_gdf, part_stops)

    # --- route along the road network (terminal A -> B through all stops) ---
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    graph, node_arr, node_list, tree = _build_network_struct(roads)
    print(f"  Road graph nodes: {len(node_list)}")

    stop_proj = part_stops.geometry.to_crs(PROJ_EPSG)
    stop_pts = np.column_stack([stop_proj.geometry.x, stop_proj.geometry.y])

    route_feats = []
    flat_rows = []
    for ri, (route, start, end) in enumerate(all_routes):
        # snap stops to nearest graph nodes
        node_of = {}
        for s in route:
            _, i = tree.query(stop_pts[s], k=1)
            node_of[s] = int(i)

        # order stops along the single shortest-path spine A -> B
        ordered, spine = _network_order(route, start, end, node_of, graph,
                                        node_list, stop_pts)
        line = LineString(spine) if len(spine) >= 2 else LineString(
            [tuple(stop_pts[start]), tuple(stop_pts[end])])
        air_km = line.length / 1000.0

        route_feats.append({
            "route_id": ri,
            "n_stops": len(ordered),
            "length_km": air_km,
            "length_km_nonlin": air_km * 2.0,
            "geometry": line,
        })
        for order, stop_idx in enumerate(ordered):
            flat_rows.append({
                "route_id": ri,
                "order": order,
                "stop_idx": stop_idx,
                "osm_id": part_stops.iloc[stop_idx]["osm_id"],
                "name": part_stops.iloc[stop_idx].get("name"),
            })

    route_gdf = gpd.GeoDataFrame(route_feats, geometry="geometry",
                                  crs=PROJ_EPSG).to_crs("EPSG:4326")
    flat = pd.DataFrame(flat_rows)

    route_gdf.to_parquet(paths["routes"])
    flat.to_parquet(paths["flat"])

    # stops_pos: only stops used by at least one route
    used_stops = flat["stop_idx"].unique()
    keep = [c for c in ("osm_id", "name", "kind", "is_terminal", "n_routes",
                        "population", "jobs", "radius_m", "geometry")
            if c in part_stops.columns]
    part_stops.iloc[used_stops][keep].to_parquet(paths["stops_pos"])

    n_served = len(used_stops)
    # --- route-count formula: O = stops in the reference file, N, S from model ---
    from src.phase3 import K1, K2, K3, calc_route_count

    n_stops_file = int(len(ref_gdf))                      # stops from the file (O)
    p1 = json.load(open(CACHE_DIR / "phase1_real" / "phase1_report.json", encoding="utf-8"))
    pop_1000 = p1["population_sum_by_stop"] / 1000.0      # N (thousand people)
    area_km2 = _area_km2()                                # S (km^2, UTM 32637)
    interchange = 13.0
    m = calc_route_count(pop_1000, area_km2, n_stops_file, interchange)
    n_routes_formula = int(round(m))

    report = {
        "n_ref_stops": len(ref_gdf),
        "n_ref_routes": len(route_info),
        "n_matched_stops": len(matched),
        "n_routes": len(all_routes),
        "n_routes_formula": n_routes_formula,
        "route_count_formula_m": round(m, 2),
        "formula": {
            "N_thousands": round(pop_1000, 1),
            "S_km2": area_km2,
            "O_stops_file": n_stops_file,
            "interchange": interchange,
            "k1": K1, "k2": K2, "k3": K3,
        },
        "n_stops_served": n_served,
        "avg_route_stops": round(float(flat.groupby("route_id").size().mean()), 1)
            if all_routes else 0,
        "avg_route_km_air": round(float(route_gdf["length_km"].mean()), 2)
            if len(route_gdf) else 0,
        "total_route_km_air": round(float(route_gdf["length_km"].sum()), 1)
            if len(route_gdf) else 0,
        "min_route_km": round(float(route_gdf["length_km"].min()), 2)
            if len(route_gdf) else 0,
        "max_route_km": round(float(route_gdf["length_km"].max()), 2)
            if len(route_gdf) else 0,
    }
    print(f"Route-count formula -> m = {m:.2f}, n_routes_formula = {n_routes_formula} "
          f"(O = {n_stops_file} stops from file)")
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _write_markdown(report)
    _write_map(route_gdf, part_stops)
    return report


def _write_markdown(report: dict) -> None:
    f = report.get("formula", {})
    lines = [
        "# Фаза 3 (реальные данные) — Маршруты из GeoJSON",
        "",
        f"- Остановок в GeoJSON: {report['n_ref_stops']}",
        f"- Маршрутов в GeoJSON: {report['n_ref_routes']}",
        f"- Остановок сопоставлено: {report['n_matched_stops']}",
        "",
        "## Число маршрутов (формула Якимова)",
        f"m = {f.get('k1')}*N/{f.get('interchange')} + {f.get('k2')}*S/{f.get('interchange')} "
        f"+ {f.get('k3')}*O/{f.get('interchange')} = **{report['route_count_formula_m']:.2f}**",
        f"→ **{report['n_routes_formula']} маршрутов**",
        f"- N (население, тыс.): {f.get('N_thousands'):,}",
        f"- S (площадь, км²): {f.get('S_km2')}",
        f"- O (остановок из файла): **{f.get('O_stops_file')}**",
        f"- I (пересадочность): {f.get('interchange')}",
        "",
        f"## Маршрутов построено по сети (≥{ROUTE_MIN_LENGTH_KM} км): **{report['n_routes']}**",
        "",
        f"- Остановок охвачено: {report['n_stops_served']}",
        f"- Среднее число остановок на маршрут: {report['avg_route_stops']}",
        f"- Средняя длина маршрута: {report['avg_route_km_air']} км",
        f"- Мин. длина: {report['min_route_km']} км",
        f"- Макс. длина: {report['max_route_km']} км",
        f"- Суммарная длина сети: {report['total_route_km_air']} км",
    ]
    (REPORT_DIR / "phase3_real_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_map(route_gdf: gpd.GeoDataFrame, stops: gpd.GeoDataFrame) -> None:
    import folium
    bbox = stops.total_bounds
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    for _, r in route_gdf.iterrows():
        folium.PolyLine(
            [[p[1], p[0]] for p in r.geometry.coords],
            color="#e41a1c", weight=2, opacity=0.6,
            popup=f"route {r['route_id']}, {r['n_stops']} stops, {r['length_km']:.1f} km",
        ).add_to(m)
    for _, s in stops.iterrows():
        pt = s.geometry.centroid
        folium.CircleMarker(
            location=[pt.y, pt.x], radius=2.5, color="#7f7f7f",
            fill=True, fillOpacity=0.5,
        ).add_to(m)
    m.save(str(REPORT_DIR / "phase3_real_map.html"))
    print(f"Map saved: {REPORT_DIR / 'phase3_real_map.html'}")


if __name__ == "__main__":
    report = run_phase3_real(force=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
