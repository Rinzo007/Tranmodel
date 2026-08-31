"""Phase 4: passenger-flow assignment (pasazhiropotok) on the route network.

Assigns the Phase-2 OD matrix onto the Phase-3 route network via a route
planning model that allows up to MAX_TRANSFERS transfers between routes.

A trip orig -> dest is planned as a sequence of "legs", each on a single route
(a route is an ordered, unidirectional list of stops). Transfers happen at
stops shared by consecutive routes. The best plan minimises the number of
transfers, then the number of traversed segments (a proxy for travel time).
Trips are then added to every segment of every leg (all-or-nothing).

Outputs (data/cache/phase4/):
  - assign_od.parquet       every OD pair + planned leg sequence
  - segment_load.parquet    load by (route_id, segment order)
  - route_load.parquet      boarding/alighting/load indicators per route
  - passenger_flow.parquet  per-stop boardings/alightings/flow
  - phase4_report.json / .md
"""

import json
from collections import defaultdict, deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config import CACHE_DIR, PROJ_EPSG, REPORT_DIR

SEG_CAPACITY = 60.0      # pass/vehicle (bus capacity used for load factor)
VEH_PER_DAY = 30.0       # total vehicle-runs per route per day (estimate)
MAX_TRANSFERS = 2        # allowed number of transfers for a trip


class Phase4Error(RuntimeError):
    pass


def load_inputs() -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    flat = pd.read_parquet(CACHE_DIR / "phase3_real" / "routes_flat.parquet")
    routes = gpd.read_parquet(CACHE_DIR / "phase3_real" / "routes.parquet")
    od = pd.read_parquet(CACHE_DIR / "phase2" / "matrix_od.parquet")
    if flat.empty or od.empty:
        raise Phase4Error("Missing routes or OD matrix (run Phases 2-3 first)")
    return flat, routes, od


def build_index(flat: pd.DataFrame) -> tuple[dict, dict]:
    """Stop->order map per route and stop->set(routes) map."""
    route_order: dict[int, dict] = {}
    stop_routes: dict[int, set[int]] = {}
    for rid, g in flat.groupby("route_id"):
        d = dict(zip(g["stop_idx"].astype(int), g["order"].astype(int)))
        route_order[int(rid)] = d
        for s in d:
            stop_routes.setdefault(int(s), set()).add(int(rid))
    return route_order, stop_routes


def build_route_graph(route_order, stop_routes) -> tuple[dict, dict]:
    """Graph over routes (nodes=route_id).

    Returns (adjacency with shared stop, all-pairs shortest fields).
    adj[r1] = {r2: a common-stop at which a transfer r1->r2 is possible}
    """
    adj: dict[int, dict] = defaultdict(dict)
    for s, rs in stop_routes.items():
        rs = sorted(rs)
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                r1, r2 = rs[a], rs[b]
                if r2 not in adj[r1]:
                    adj[r1][r2] = s
                if r1 not in adj[r2]:
                    adj[r2][r1] = s
    return dict(adj)


def _bfs_routes(src, adj):
    """BFS distances from route src: returns dist dict, parent dict, entry-stop dict."""
    dist = {src: 0}
    parent = {src: None}
    entry = {src: None}   # stop used to step from parent -> node
    q = deque([src])
    while q:
        r = q.popleft()
        for nxt, s in adj.get(r, {}).items():
            if nxt not in dist:
                dist[nxt] = dist[r] + 1
                parent[nxt] = r
                entry[nxt] = s
                q.append(nxt)
    return dist, parent, entry


def _plan_legs(orig, dest, Mi, Mj, route_order, dist_all, parent_all, entry_all):
    """Return best (legs, n_transfers, total_steps) or None.

    legs = list of (route_id, board_stop, alight_stop, board_order, alight_order).
    """
    best = None
    for r in Mi:
        if r not in dist_all:
            continue
        dist, parent, entry = dist_all[r], parent_all[r], entry_all[r]
        for rd in Mj:
            if rd not in dist:
                continue
            transfers = dist[rd]
            if transfers > MAX_TRANSFERS:
                continue
            # reconstruct route chain
            chain = []
            cur = rd
            while cur is not None:
                chain.append(cur)
                cur = parent[cur]
            chain.reverse()
            # common stops between consecutive routes
            common = {}
            for a in range(len(chain) - 1):
                common[(chain[a], chain[a + 1])] = entry[chain[a + 1]]
            # build legs only when direction is valid on every route
            legs, ok = _materialize(orig, dest, chain, common, route_order)
            if not ok:
                continue
            steps = sum((a - b) for _, _, _, b, a in legs)
            if best is None or (transfers, steps) < (best[1], best[2]):
                best = (legs, transfers, steps)
    return best


def _materialize(orig, dest, chain, common, route_order):
    """Turn route chain into concrete legs board_stop->alight_stop per route."""
    legs = []
    # determine entry stop into the very first route = orig
    # leg on chain[k] goes from 'in_stop' to 'out_stop'
    in_stop = orig
    for k in range(len(chain)):
        r = chain[k]
        out_stop = dest if k == len(chain) - 1 else common[(chain[k], chain[k + 1])]
        o = route_order[r].get(in_stop)
        d = route_order[r].get(out_stop)
        if o is None or d is None or o >= d:
            return [], False
        legs.append((r, in_stop, out_stop, o, d))
        in_stop = out_stop  # board next route at this stop
    return legs, True


class RoutePlanner:
    """Precomputes BFS over the route graph once and answers OD queries."""

    def __init__(self, route_order, stop_routes):
        self.route_order = route_order
        self.stop_routes = stop_routes
        adj = build_route_graph(route_order, stop_routes)
        self.dist_all = {}
        self.parent_all = {}
        self.entry_all = {}
        for src in adj:
            d, p, e = _bfs_routes(src, adj)
            self.dist_all[src] = d
            self.parent_all[src] = p
            self.entry_all[src] = e

    def plan(self, orig: int, dest: int):
        Mi = self.stop_routes.get(orig)
        Mj = self.stop_routes.get(dest)
        if not Mi or not Mj:
            return None
        return _plan_legs(orig, dest, Mi, Mj, self.route_order,
                          self.dist_all, self.parent_all, self.entry_all)


def assign_od(od, planner) -> tuple[pd.DataFrame, dict]:
    """Plan every OD pair (up to MAX_TRANSFERS). Returns (assign_df, stats)."""
    rows = []
    assigned = 0.0
    n_tr = [0, 0, 0, 0]  # count by number of transfers (0,1,2,>2/unserved)

    for orig, dest, trips in zip(od["orig"], od["dest"], od["trips"]):
        plan = planner.plan(int(orig), int(dest))
        if plan is None:
            n_tr[3] += 1
            continue
        legs, transfers, steps = plan
        leg_list = [
            {"route_id": r, "board_stop": b, "alight_stop": a,
             "board_order": bo, "alight_order": ao}
            for (r, b, a, bo, ao) in legs
        ]
        n_tr[transfers] += 1
        assigned += float(trips)
        rows.append({
            "orig": int(orig), "dest": int(dest), "trips": float(trips),
            "n_transfers": transfers, "total_steps": steps,
            "legs": leg_list,
        })

    assign = pd.DataFrame(rows)
    if assign.empty:
        assign = pd.DataFrame(columns=[
            "orig", "dest", "trips", "n_transfers", "total_steps", "legs"])
    else:
        assign = assign.sort_values("trips", ascending=False).reset_index(drop=True)

    total = float(od["trips"].sum())
    stats = {
        "od_pairs": int(len(od)),
        "od_pairs_assigned": int(len(assign)),
        "total_trips": round(total, 1),
        "assigned_trips": round(assigned, 1),
        "assigned_share": round(assigned / total, 4) if total > 0 else 0.0,
        "n_direct": int(n_tr[0]),
        "n_1transfer": int(n_tr[1]),
        "n_2transfers": int(n_tr[2]),
        "n_unserved": int(n_tr[3]),
    }
    if not assign.empty:
        stats["avg_transfers"] = round(float(assign["n_transfers"].mean()), 3)
    else:
        stats["avg_transfers"] = 0.0
    return assign, stats


def segment_loads(assign: pd.DataFrame) -> pd.DataFrame:
    """Accumulate passenger load per (route_id, segment order) over all legs."""
    seg: dict[tuple[int, int], float] = {}
    for row in assign.itertuples(index=False):
        trips = row.trips
        for leg in row.legs:
            rid = int(leg["route_id"])
            for k in range(int(leg["board_order"]), int(leg["alight_order"])):
                key = (rid, k)
                seg[key] = seg.get(key, 0.0) + trips
    return pd.DataFrame(
        [{"route_id": r, "seg_order": k, "load": v} for (r, k), v in seg.items()])


def passenger_flow_summary(assign: pd.DataFrame) -> pd.DataFrame:
    """Per-stop ridership (board+alight) across the network."""
    board = defaultdict(float)
    alight = defaultdict(float)
    for row in assign.itertuples(index=False):
        trips = row.trips
        for leg in row.legs:
            board[int(leg["board_stop"])] += trips
            alight[int(leg["alight_stop"])] += trips
    out = pd.DataFrame({"stop_idx": list(set(board) | set(alight))})
    out["boardings"] = out["stop_idx"].map(board).fillna(0.0)
    out["alightings"] = out["stop_idx"].map(alight).fillna(0.0)
    out["flow"] = out["boardings"] + out["alightings"]
    return out.sort_values("flow", ascending=False).reset_index(drop=True)


def run_phase4(force: bool = False) -> dict:
    out_dir = CACHE_DIR / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "seg": out_dir / "segment_load.parquet",
        "route": out_dir / "route_load.parquet",
        "assign": out_dir / "assign_od.parquet",
        "flow": out_dir / "passenger_flow.parquet",
        "report": out_dir / "phase4_report.json",
    }
    if all(p.exists() for p in paths.values()) and not force:
        return json.load(open(paths["report"], encoding="utf-8"))

    flat, routes, od = load_inputs()
    route_order, stop_routes = build_index(flat)

    print(f"Routes: {flat['route_id'].nunique()}, OD pairs: {len(od)}")
    planner = RoutePlanner(route_order, stop_routes)
    assign, stats = assign_od(od, planner)
    seg = segment_loads(assign)
    flow = passenger_flow_summary(assign)

    seg.to_parquet(paths["seg"])
    assign.to_parquet(paths["assign"])
    flow.to_parquet(paths["flow"])

    # route-level indicators
    r_load = seg.groupby("route_id")["load"].agg(
        max_load="max", mean_load="mean", sum_load="sum").reset_index()
    r_sz = flat.groupby("route_id").size().rename("n_stops").reset_index()
    r_df = r_sz.merge(r_load, on="route_id", how="left").fillna(0.0)
    r_df["load_factor"] = r_df["max_load"] / (SEG_CAPACITY * VEH_PER_DAY)
    r_df = r_df.merge(routes.drop(columns="geometry"), on="route_id", how="left")
    r_df = r_df.sort_values("max_load", ascending=False).reset_index(drop=True)
    r_df.to_parquet(paths["route"])

    report = {}
    report.update(stats)
    report.update({
        "n_routes": int(flat["route_id"].nunique()),
        "n_segments": int(len(seg)),
        "max_segment_load": round(float(seg["load"].max()), 1) if len(seg) else 0.0,
        "top_route_id": int(r_df.iloc[0]["route_id"]) if len(r_df) else None,
        "top_route_max_load": round(float(r_df.iloc[0]["max_load"]), 1) if len(r_df) else 0.0,
        "avg_route_max_load": round(float(r_df["max_load"].mean()), 1) if len(r_df) else 0.0,
        "avg_load_factor": round(float(r_df["load_factor"].mean()), 3) if len(r_df) else 0.0,
        "seg_capacity": SEG_CAPACITY,
        "veh_per_day": VEH_PER_DAY,
        "top_stop_idx": int(flow.iloc[0]["stop_idx"]) if len(flow) else None,
        "top_stop_flow": round(float(flow.iloc[0]["flow"]), 1) if len(flow) else 0.0,
    })
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_markdown(report)
    _write_map(seg, flat)
    return report


def _write_markdown(report: dict) -> None:
    lines = [
        "# Фаза 4 — Пассажиропоток на маршрутной сети",
        "",
        f"- Число маршрутов: {report['n_routes']}",
        f"- Пар корреспонденций (OD): {report['od_pairs']:,}",
        f"- OD-пар, обслуживаемых напрямую (без пересадки): {report['od_pairs_assigned']:,}",
        f"- Объём поездок по матрице: {report['total_trips']:,.0f}",
        f"- Поездок распределено на сеть: {report['assigned_trips']:,.0f} "
        f"({report['assigned_share']*100:.1f}%)",
        "",
        f"- Максимальная загрузка перегона: **{report['max_segment_load']:,.0f}**",
        f"- Средняя максимальная загрузка маршрута: {report['avg_route_max_load']:,.0f}",
        f"- Средний коэффициент заполнения (макс. по маршруту): {report['avg_load_factor']:.3f}",
        f"  (ёмкость = {report['seg_capacity']:.0f} чел., рейсов в день = {report['veh_per_day']:.0f})",
        f"- Наиболее загруженная остановка (посадки+высадки): "
        f"{report['top_stop_idx'] if report['top_stop_idx'] is not None else '-'} "
        f"({report['top_stop_flow']:,.0f})",
        "",
    ]
    (REPORT_DIR / "phase4_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_map(seg: pd.DataFrame, flat: pd.DataFrame) -> None:
    import folium
    routes = gpd.read_parquet(CACHE_DIR / "phase3_real" / "routes.parquet")

    seg_max = float(seg["load"].max()) if len(seg) else 1.0
    look = dict(zip(zip(seg["route_id"], seg["seg_order"]), seg["load"])) \
        if len(seg) else {}

    m = folium.Map(location=[51.66, 39.2], zoom_start=11, tiles="CartoDB positron")
    colormap = folium.LinearColormap(["#440154", "#3b528b", "#21918c", "#5ec962",
                                      "#fde725"], vmin=0, vmax=seg_max,
                                     caption="Passenger load per segment")

    for _, r in routes.iterrows():
        coords = list(r.geometry.coords) if r.geometry is not None else []
        if len(coords) < 2:
            continue
        for k in range(len(coords) - 1):
            load = look.get((int(r["route_id"]), k), 0.0)
            folium.PolyLine(
                [[coords[k][1], coords[k][0]], [coords[k + 1][1], coords[k + 1][0]]],
                color=colormap(load), weight=3.0, opacity=0.85,
            ).add_to(m)
    colormap.add_to(m)
    m.save(str(REPORT_DIR / "phase4_map.html"))
    print(f"Map saved: {REPORT_DIR / 'phase4_map.html'}")


if __name__ == "__main__":
    report = run_phase4()
    print(json.dumps(report, indent=2, ensure_ascii=False))
