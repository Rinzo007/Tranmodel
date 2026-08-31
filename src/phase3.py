"""Phase 3: generate a public-transport route network.

Implements the route-count formula and the route-building algorithm from
M. R. Yakimov, "Approaches to formation of an effective route network of large
cities" (2022).

Route count:  m = k1*N/I + k2*S/I + k3*O/I
  k1=0.156, k2=0.729, k3=0.375
  N - population (thousand), S - urban area (km^2), O - number of stops,
  I - interchange coefficient.

Route building (greedy, one route at a time):
  - a route starts from a terminal stop;
  - candidate next stops are active stops within a search radius (start 500 m,
    grow by 200 m up to 5000 m);
  - for every candidate, the passenger-flow on the resulting segment is
    computed as the number of "rider" correspondences (those whose destination
    lies ahead of the candidate along the route);
  - the candidate giving the maximum on-route flow is appended;
  - stops whose distance (from the previous stop) would increase the route
    length are deactivated, keeping the route near-straight;
  - the route terminates when no active stop is within range or max length/_stops
    reached.

Outputs (data/cache/phase3/):
  - routes.parquet        each route as a GeoDataFrame (geometry = polyline)
  - routes_flat.parquet   long-form stops per route
  - phase3_report.json / .md
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.spatial as spatial
from shapely.geometry import LineString

from config import CACHE_DIR, LAYERS_DIR, REPORT_DIR, PROJ_EPSG

# --- route-count coefficients (Yakimov 2022, eq. 4) ---
K1, K2, K3 = 0.156, 0.729, 0.375
# --- route-building parameters ---
SEARCH_RADIUS_START_M = 500.0
SEARCH_RADIUS_STEP_M = 200.0
SEARCH_RADIUS_MAX_M = 5000.0
ROUTE_MAX_LENGTH_KM = 20.0
# non-linearity coefficient: network distance ~= K_NONLIN * air distance
K_NONLIN = 2.0


class Phase3Error(RuntimeError):
    pass


def calc_route_count(
    population_thousands: float,
    area_km2: float,
    n_stops: int,
    interchange: float = 1.15,
) -> float:
    """m = k1*N/I + k2*S/I + k3*O/I (N in thousands, S in km^2)."""
    if interchange <= 0:
        raise Phase3Error("Interchange coefficient must be > 0")
    m = (K1 * population_thousands + K2 * area_km2 + K3 * n_stops) / interchange
    return m


def _build_cost_struct(stops: gpd.GeoDataFrame):
    """Projected point array (km) for distance computations + KDTree."""
    stop_proj = stops.geometry.to_crs(PROJ_EPSG)
    pts = np.column_stack([stop_proj.geometry.x / 1000.0,
                           stop_proj.geometry.y / 1000.0])
    tree = spatial.cKDTree(pts)
    return pts, tree


def _candidate_stops(
    tree: spatial.cKDTree,
    pts: np.ndarray,
    current_idx: int,
    n_stops: int,
    active: np.ndarray,
    radius: float,
) -> list[int]:
    """Active stops within `radius` (m) of the current stop, excluding self."""
    dist, idx = tree.query(pts[current_idx], k=min(64, n_stops),
                           distance_upper_bound=radius / 1000.0)
    cands = []
    for d, i in zip(dist, idx):
        if i >= n_stops or i == current_idx:
            continue
        if not active[i]:
            continue
        cands.append(int(i))
    return cands


def _generate_route(
    start: int,
    active: np.ndarray,
    tree: spatial.cKDTree,
    pts: np.ndarray,
    T: np.ndarray,
    stops_idx: list,
    dist_from_first: np.ndarray,
    n: int,
) -> tuple[list[int], float]:
    """Grow one route from `start`. Returns (route_stops, route_len_km).

    On-board flow is computed vectorized: S = trips from all on-route origins to
    every destination; for a candidate, flow = sum of S over active destinations
    further from the route start than the candidate (kept "ahead").
    """
    route = [start]
    active[start] = False
    route_len_km = 0.0

    while route_len_km < ROUTE_MAX_LENGTH_KM:
        radius = SEARCH_RADIUS_START_M
        cands = []
        while radius <= SEARCH_RADIUS_MAX_M:
            cands = _candidate_stops(tree, pts, route[-1], n, active, radius)
            if cands:
                break
            radius += SEARCH_RADIUS_STEP_M
        if not cands:
            break

        # aggregate trips from all on-route origins (vector over destinations)
        S = np.zeros(n)
        for orig_pos in route:
            S += T[stops_idx[orig_pos], :]

        # forbid revisiting stops already on this route
        on_route = np.zeros(n, dtype=bool)
        on_route[route] = True

        best = None
        best_flow = -1.0
        for c in cands:
            if on_route[c]:
                continue
            d_c = dist_from_first[c]
            # destinations strictly ahead of candidate, active & not on route
            keep = active & ~on_route & (dist_from_first > d_c)
            flow = float(S[keep].sum())
            if flow > best_flow:
                best_flow = flow
                best = c
        if best is None:
            break

        d_air = float(np.hypot(pts[best][0] - pts[route[-1]][0],
                               pts[best][1] - pts[route[-1]][1]))
        if route_len_km + d_air * K_NONLIN > ROUTE_MAX_LENGTH_KM:
            break
        route_len_km += d_air * K_NONLIN
        route.append(best)
    return route, route_len_km


def generate_routes(
    stops: gpd.GeoDataFrame,
    T: np.ndarray,
    stops_idx: list,
    n_routes: int,
    seed_stops: list[int] | None = None,
) -> list[list[int]]:
    """Generate `n_routes` routes as lists of stop positions in `stops` order.

    Greedy re-synthesis: after a route is built, its stops are deactivated so
    subsequent routes target under-served areas. A route grows by repeatedly
    taking the candidate stop (within search radius) that maximizes on-board
    passenger flow to destinations ahead.
    """
    n = len(stops)
    pts, tree = _build_cost_struct(stops)
    first = pts[0]
    dist_from_first = np.sqrt(((pts - first) ** 2).sum(axis=1))

    active = np.ones(n, dtype=bool)

    if seed_stops is None:
        prods = stops["population"].to_numpy()
        order = np.argsort(-prods)
        seed_stops = [int(i) for i in order]

    routes: list[list[int]] = []
    seed_iter = iter(seed_stops)

    for _ in range(n_routes):
        start = None
        for s in seed_iter:
            if active[s]:
                start = s
                break
        if start is None:
            remaining = np.flatnonzero(active)
            if len(remaining) == 0:
                break
            start = int(remaining[0])

        # a stop already used as start can still be re-visited along a route,
        # so mark it inactive only as a seed for subsequent new routes
        route, _len = _generate_route(start, active, tree, pts, T, stops_idx,
                                      dist_from_first, n)
        routes.append(route)
    return routes


def run_phase3(force: bool = False) -> dict:
    out_dir = CACHE_DIR / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "routes": out_dir / "routes.parquet",
        "flat": out_dir / "routes_flat.parquet",
        "report": out_dir / "phase3_report.json",
        "stops_pos": out_dir / "stops_pos.parquet",
    }
    if all(p.exists() for p in paths.values()) and not force:
        return json.load(open(paths["report"], encoding="utf-8"))

    # stops that participate
    stops = gpd.read_parquet(CACHE_DIR / "phase1" / "stops_demand.parquet")
    stops = stops[stops["radius_m"].notna()].reset_index(drop=True)
    n = len(stops)
    if n == 0:
        raise Phase3Error("No participating stops (run Phase 1)")

    # boundary area (from stored boundary) for the route-count formula
    area_km2 = _area_km2()

    population_thousands = stops["population"].sum() / 1000.0
    n_stops = n
    interchange = 13.0
    m = calc_route_count(population_thousands, area_km2, n_stops, interchange)
    n_routes = int(round(m))

    print(f"Route count formula -> m = {m:.1f}, using n_routes = {n_routes}")

    # OD matrix
    od = pd.read_parquet(CACHE_DIR / "phase2" / "matrix_od.parquet")
    stops_idx = list(stops.index)
    pos = {s: i for i, s in enumerate(stops_idx)}
    T = np.zeros((n, n))
    o = od["orig"].map(pos)
    d = od["dest"].map(pos)
    mask = o.notna() & d.notna()
    np.add.at(T, (o[mask].astype(int), d[mask].astype(int)), od.loc[mask, "trips"].to_numpy())

    routes = generate_routes(stops, T, stops_idx, n_routes)

    # build geometry
    stop_proj = stops.geometry.to_crs(PROJ_EPSG)
    route_feats = []
    flat_rows = []
    for ri, route in enumerate(routes):
        if len(route) < 2:
            continue
        coords = [(stop_proj.geometry.iloc[stop_idx].x,
                   stop_proj.geometry.iloc[stop_idx].y) for stop_idx in route]
        line = LineString(coords)
        air_km = line.length / 1000.0
        route_feats.append({"route_id": ri, "n_stops": len(route),
                            "length_km": air_km,
                            "length_km_nonlin": air_km * K_NONLIN,
                            "geometry": line})
        for order, stop_idx in enumerate(route):
            flat_rows.append({"route_id": ri, "order": order,
                              "stop_idx": stop_idx,
                              "osm_id": stops.iloc[stop_idx]["osm_id"],
                              "name": stops.iloc[stop_idx].get("name")})

    route_gdf = gpd.GeoDataFrame(route_feats, geometry="geometry", crs=PROJ_EPSG).to_crs("EPSG:4326")
    flat = pd.DataFrame(flat_rows)

    route_gdf.to_parquet(paths["routes"])
    flat.to_parquet(paths["flat"])
    stops[["osm_id", "name", "kind", "railway", "highway", "population", "jobs", "radius_m", "geometry"]].to_parquet(paths["stops_pos"])

    n_served = int(flat["stop_idx"].nunique())
    report = {
        "n_stops_total": n,
        "population_thousands": round(population_thousands, 1),
        "area_km2": round(area_km2, 1),
        "interchange": interchange,
        "k1": K1, "k2": K2, "k3": K3,
        "k_nonlin": K_NONLIN,
        "route_count_formula_m": round(m, 2),
        "n_routes": n_routes,
        "n_routes_built": len(routes),
        "stops_served": n_served,
        "avg_route_stops": round(float(flat.groupby("route_id").size().mean()), 2) if len(routes) else 0,
        "avg_route_km_air": round(float(route_gdf["length_km"].mean()), 2) if len(routes) else 0,
        "avg_route_km_nonlin": round(float(route_gdf["length_km_nonlin"].mean()), 2) if len(routes) else 0,
        "total_route_km_air": round(float(route_gdf["length_km"].sum()), 1),
        "total_route_km_nonlin": round(float(route_gdf["length_km_nonlin"].sum()), 1),
    }
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_markdown(report)
    _write_map(route_gdf, stops)
    return report


def _area_km2() -> float:
    from src.boundary import load_boundary
    from config import NAMES, PROJ_EPSG
    b = load_boundary(CACHE_DIR / f"{NAMES['boundary']}.geojson")
    return float(b.to_crs(PROJ_EPSG).area.iloc[0] / 1e6)


def _write_markdown(report: dict) -> None:
    lines = [
        "# Фаза 3 — Генерация маршрутов",
        "",
        f"- Остановок (участвующих): {report['n_stops_total']}",
        f"- Население (тыс.): {report['population_thousands']}",
        f"- Площадь города (км²): {report['area_km2']}",
        f"- Коэффициент пересадочности I: {report['interchange']}",
        f"- Коэффициент непрямолинейности: **{report['k_nonlin']}**",
        "",
        f"## Число маршрутов: **{report['n_routes']}**",
        f"m = {report['k1']}*{round(report['population_thousands'],1)}/{report['interchange']} "
        f"+ {report['k2']}*{round(report['area_km2'],1)}/{report['interchange']} "
        f"+ {report['k3']}*{report['n_stops_total']}/{report['interchange']} = {report['route_count_formula_m']:.2f}",
        "",
        f"- Построено маршрутов: {report['n_routes_built']}",
        f"- Остановок охвачено: {report['stops_served']}",
        f"- Среднее число остановок на маршрут: {report['avg_route_stops']}",
        f"- Средняя длина маршрута (по воздушной): {report['avg_route_km_air']} км",
        f"- Средняя длина маршрута (с учётом непрямолинейности): {report['avg_route_km_nonlin']} км",
        f"- Суммарная длина сети (по воздушной): {report['total_route_km_air']} км",
        f"- Суммарная длина сети (с учётом непрямолинейности): {report['total_route_km_nonlin']} км",
        "",
    ]
    (REPORT_DIR / "phase3_report.md").write_text("\n".join(lines), encoding="utf-8")


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
        folium.CircleMarker(location=[pt.y, pt.x], radius=2.5, color="#7f7f7f",
                            fill=True, fillOpacity=0.5).add_to(m)
    m.save(str(REPORT_DIR / "phase3_map.html"))
    print(f"Map saved: {REPORT_DIR / 'phase3_map.html'}")


if __name__ == "__main__":
    report = run_phase3()
    print(json.dumps(report, indent=2, ensure_ascii=False))