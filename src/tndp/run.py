"""End-to-end TNDP route synthesis from zone-based OD demand."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from config import CACHE_DIR, LAYERS_DIR, REPORT_DIR, PROJ_EPSG
from src.aequilibrae_pipeline import build_project
from src.zone_od import run_zone_od
from .aequilibrae_eval import evaluate_route_set_aequilibrae
from .candidates import generate_route_candidates
from .corridors import DemandCorridor, extract_demand_corridors
from .export import routes_to_geojson
from .io import save_route_set
from .model import Evaluation, NetworkDesignConfig, RouteSet
from .network import add_stop_nodes, build_tndp_graph, snap_stops_to_graph
from .optimizer import TNDPOptimizer
from .path_cache import build_stop_path_index
from .surrogate import surrogate_evaluator
from .zone_io import load_zone_demand

OUTPUT_DIR = CACHE_DIR / "tndp"
EVAL_CACHE = OUTPUT_DIR / "evaluation_cache"
PATH_CACHE = OUTPUT_DIR / "stop_paths.pkl"


def _load_inputs():
    demand, zones = load_zone_demand()
    stops = gpd.read_parquet(CACHE_DIR / "phase1_real" / "stops_demand.parquet").to_crs(PROJ_EPSG).reset_index(drop=True)
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    road_graph = build_tndp_graph(roads)
    _, stop_mapping, _ = snap_stops_to_graph(road_graph, stops)
    stop_graph = add_stop_nodes(road_graph, stop_mapping, k_neighbors=8)
    stop_xy = np.column_stack([stops.geometry.x.to_numpy(dtype=float), stops.geometry.y.to_numpy(dtype=float)]) / 1000.0
    zone_points = zones.geometry.centroid
    zone_xy = np.column_stack([zone_points.x.to_numpy(dtype=float), zone_points.y.to_numpy(dtype=float)]) / 1000.0
    # Keep several nearest stops per zone for future demand assignment; the first
    # remains the deterministic fast-screening representative.
    zone_tree = cKDTree(stop_xy)
    _, zone_to_stops = zone_tree.query(zone_xy, k=min(4, len(stops)))
    zone_to_stops = np.atleast_2d(zone_to_stops)
    if zone_to_stops.shape[0] != len(zones):
        zone_to_stops = zone_to_stops.T
    zone_to_stop = zone_to_stops[:, 0].astype(int)
    terminal_nodes = set(np.flatnonzero(stops["is_terminal"].fillna(False).to_numpy()).tolist())
    stop_demand = np.zeros(len(stops), dtype=float)
    zone_mass = zones["production"].to_numpy(dtype=float) + zones["attraction"].to_numpy(dtype=float)
    np.add.at(stop_demand, zone_to_stop, zone_mass)
    stop_lonlat = stops.to_crs("EPSG:4326")
    stop_lonlat_xy = np.column_stack([stop_lonlat.geometry.x.to_numpy(dtype=float), stop_lonlat.geometry.y.to_numpy(dtype=float)])
    return demand, zones, stops, road_graph, stop_graph, stop_mapping, stop_xy, zone_xy, zone_to_stop, zone_to_stops, stop_demand, stop_lonlat_xy, terminal_nodes


def _map_corridors_to_stops(corridors, zone_to_stop):
    out, seen = [], set()
    for corridor in corridors:
        origin, destination = int(zone_to_stop[corridor.origin]), int(zone_to_stop[corridor.destination])
        if origin == destination or (origin, destination) in seen:
            continue
        seen.add((origin, destination))
        out.append(DemandCorridor(origin, destination, corridor.demand, corridor.direct_distance_km))
    return out


def _empty_evaluation(demand, config):
    total = float(np.asarray(demand, dtype=float).sum())
    return Evaluation(score=total * config.uncovered_demand_weight, uncovered_demand=total,
                      direct_demand_share=0.0, metadata={"evaluator": "empty-network baseline"})


def run_tndp(config=None, *, full_assignment=True, progress=None):
    config = config or NetworkDesignConfig()
    config.validate()
    notify = progress or (lambda msg: print(f"[TNDP] {msg}", flush=True))
    if not (CACHE_DIR / "zone_od" / "od_matrix.parquet").exists():
        notify("Подготавливаем матрицу корреспонденций...")
        run_zone_od(zone_size_m=750.0, force=False)

    notify("Загружаем зоны, остановки и реальный дорожный граф...")
    demand, zones, stops, road_graph, stop_graph, stop_mapping, stop_xy, zone_xy, zone_to_stop, zone_to_stops, stop_demand, stop_lonlat_xy, terminal_nodes = _load_inputs()
    notify(f"Загружено: {len(zones)} зон, {len(stops)} остановок, {road_graph.number_of_nodes():,} узлов дорог")
    zone_corridors = extract_demand_corridors(demand, zone_xy, top_pairs=config.corridor_top_pairs, max_distance_km=config.corridor_distance_km)
    corridors = _map_corridors_to_stops(zone_corridors, zone_to_stop)
    notify(f"Выделено {len(corridors)} OD-коридоров")
    candidates = generate_route_candidates(corridors, stop_graph, stop_xy, node_ids=list(range(len(stops))), demand_vector=stop_demand, terminal_nodes=terminal_nodes, config=config)
    if not candidates:
        raise RuntimeError("TNDP generated no feasible route candidates")
    notify(f"Сгенерировано {len(candidates)} кандидатных маршрутов")

    # Precompute only stop pairs actually occurring in candidate routes.
    stop_pairs = {(a, b) for route in candidates for a, b in zip(route.nodes[:-1], route.nodes[1:])}
    path_index = build_stop_path_index(road_graph, stop_mapping, stop_pairs, PATH_CACHE)
    notify(f"Кэш путей остановка→остановка: {len(path_index.paths):,} сегментов")

    screened = sorted(((surrogate_evaluator(demand, zone_xy, RouteSet([r]), config, zone_to_stop, stop_xy).score, r) for r in candidates), key=lambda x: x[0])
    shortlist = [r for _, r in screened[:min(len(screened), max(config.min_routes * 3, 24))]]
    notify(f"Быстрый отбор завершён: {len(shortlist)} кандидатов перед TNDP")

    if full_assignment:
        notify("Подготавливаем минимальный Transit-проект AequilibraE...")
        project_path = build_project(force=False, progress=notify, mode="transit")
        notify("Transit-проект AequilibraE готов. Запускаем оптимизацию маршрутной сети...")
        def evaluator(route_set):
            if not route_set.route_count():
                return _empty_evaluation(demand, config)
            return evaluate_route_set_aequilibrae(route_set, demand, stop_lonlat_xy, project_path, config,
                                                  road_graph=road_graph, stop_mapping=stop_mapping,
                                                  path_index=path_index, cache_dir=EVAL_CACHE)
        fast_evaluator = lambda rs: surrogate_evaluator(demand, zone_xy, rs, config, zone_to_stop, stop_xy)
    else:
        evaluator = lambda rs: surrogate_evaluator(demand, zone_xy, rs, config, zone_to_stop, stop_xy)
        fast_evaluator = evaluator

    result = TNDPOptimizer(shortlist, evaluator, config, fast_evaluator=fast_evaluator, progress=progress).solve(graph=stop_graph)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    route_path = save_route_set(result.routes, OUTPUT_DIR / "generated_routes.json")
    geojson_path = routes_to_geojson(result.routes, stops, OUTPUT_DIR / "generated_routes.geojson", road_graph=road_graph, stop_to_road_node=stop_mapping)
    (OUTPUT_DIR / "history.json").write_text(json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8")
    ev = result.evaluation
    report = {
        "backend": "Tranmodel TNDP solver", "demand_units": "transport zones", "transit_units": "transit stops", "network_units": "real road graph",
        "n_zones": int(len(zones)), "n_stops": int(len(stops)), "n_terminals": int(len(terminal_nodes)), "n_corridors": int(len(zone_corridors)),
        "n_candidates": int(len(candidates)), "n_screened_candidates": int(len(shortlist)), "n_routes": int(result.routes.route_count()),
        "score": float(ev.score), "user_cost": float(ev.user_cost), "direct_demand_share": float(ev.direct_demand_share), "uncovered_demand": float(ev.uncovered_demand),
        "transfers": float(ev.transfers), "operator_route_km": float(ev.operator_cost), "capacity_excess": float(ev.capacity_excess),
        "route_set": str(route_path), "route_geojson": str(geojson_path), "evaluator": ev.metadata.get("evaluator", "unknown"), "full_assignment": bool(full_assignment),
    }
    (OUTPUT_DIR / "tndp_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    notify(f"Синтез завершён: {result.routes.route_count()} маршрутов")
    return report
