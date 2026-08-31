"""End-to-end TNDP route synthesis from zone-based OD demand."""

from __future__ import annotations

import json

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
from .surrogate import surrogate_evaluator
from .zone_io import load_zone_demand

OUTPUT_DIR = CACHE_DIR / "tndp"
EVAL_CACHE = OUTPUT_DIR / "evaluation_cache"


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
    zone_to_stop = cKDTree(stop_xy).query(zone_xy, k=1)[1].astype(int)
    terminal_nodes = set(np.flatnonzero(stops["is_terminal"].fillna(False).to_numpy()).tolist())
    stop_demand = np.zeros(len(stops), dtype=float)
    np.add.at(stop_demand, zone_to_stop, zones["production"].to_numpy(dtype=float) + zones["attraction"].to_numpy(dtype=float))
    stop_lonlat = stops.to_crs("EPSG:4326")
    stop_lonlat_xy = np.column_stack([stop_lonlat.geometry.x.to_numpy(dtype=float), stop_lonlat.geometry.y.to_numpy(dtype=float)])
    return demand, zones, stops, road_graph, stop_graph, stop_mapping, stop_xy, zone_xy, zone_to_stop, stop_demand, stop_lonlat_xy, terminal_nodes


def _map_corridors_to_stops(corridors: list[DemandCorridor], zone_to_stop: np.ndarray) -> list[DemandCorridor]:
    out, seen = [], set()
    for corridor in corridors:
        origin, destination = int(zone_to_stop[corridor.origin]), int(zone_to_stop[corridor.destination])
        if origin == destination or (origin, destination) in seen:
            continue
        seen.add((origin, destination))
        out.append(DemandCorridor(origin, destination, corridor.demand, corridor.direct_distance_km))
    return out


def _empty_evaluation(demand: np.ndarray, config: NetworkDesignConfig) -> Evaluation:
    total = float(demand.sum())
    return Evaluation(score=total * config.uncovered_demand_weight, uncovered_demand=total,
                      direct_demand_share=0.0, metadata={"evaluator": "empty-network baseline"})


def run_tndp(config: NetworkDesignConfig | None = None, *, full_assignment: bool = True) -> dict:
    """Synthesize a transit route network from independent transport-zone OD demand."""
    config = config or NetworkDesignConfig()
    config.validate()
    if not (CACHE_DIR / "zone_od" / "od_matrix.parquet").exists():
        run_zone_od(zone_size_m=750.0, force=False)

    demand, zones, stops, road_graph, stop_graph, stop_mapping, stop_xy, zone_xy, zone_to_stop, stop_demand, stop_lonlat_xy, terminal_nodes = _load_inputs()
    zone_corridors = extract_demand_corridors(demand, zone_xy, top_pairs=config.corridor_top_pairs,
                                               max_distance_km=config.corridor_distance_km)
    corridors = _map_corridors_to_stops(zone_corridors, zone_to_stop)
    candidates = generate_route_candidates(corridors, stop_graph, stop_xy, node_ids=list(range(len(stops))),
                                           demand_vector=stop_demand, terminal_nodes=terminal_nodes, config=config)
    if not candidates:
        raise RuntimeError("TNDP generated no feasible route candidates")

    screened = sorted(((surrogate_evaluator(demand, zone_xy, RouteSet([r]), config, zone_to_stop, stop_xy).score, r)
                       for r in candidates), key=lambda x: x[0])
    shortlist = [r for _, r in screened[:min(len(screened), max(config.min_routes * 3, 24))]]

    if full_assignment:
        project_path = build_project(force=False)

        def evaluator(route_set: RouteSet) -> Evaluation:
            if not route_set.route_count():
                return _empty_evaluation(demand, config)
            return evaluate_route_set_aequilibrae(
                route_set,
                demand,
                stop_lonlat_xy,
                project_path,
                config,
                road_graph=road_graph,
                stop_mapping=stop_mapping,
                cache_dir=EVAL_CACHE,
            )
    else:
        evaluator = lambda route_set: surrogate_evaluator(demand, zone_xy, route_set, config, zone_to_stop, stop_xy)

    result = TNDPOptimizer(shortlist, evaluator, config).solve(graph=stop_graph)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    route_path = save_route_set(result.routes, OUTPUT_DIR / "generated_routes.json")
    geojson_path = routes_to_geojson(result.routes, stops, OUTPUT_DIR / "generated_routes.geojson", road_graph=road_graph, stop_to_road_node=stop_mapping)
    (OUTPUT_DIR / "history.json").write_text(json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8")

    ev = result.evaluation
    report = {
        "backend": "Tranmodel TNDP solver",
        "demand_units": "transport zones",
        "transit_units": "transit stops",
        "network_units": "real road graph",
        "n_zones": int(len(zones)), "n_stops": int(len(stops)), "n_terminals": int(len(terminal_nodes)),
        "n_corridors": int(len(zone_corridors)), "n_candidates": int(len(candidates)),
        "n_screened_candidates": int(len(shortlist)), "n_routes": int(result.routes.route_count()),
        "score": float(ev.score), "user_cost": float(ev.user_cost),
        "direct_demand_share": float(ev.direct_demand_share), "uncovered_demand": float(ev.uncovered_demand),
        "transfers": float(ev.transfers), "operator_route_km": float(ev.operator_cost),
        "capacity_excess": float(ev.capacity_excess), "route_set": str(route_path), "route_geojson": str(geojson_path),
        "evaluator": ev.metadata.get("evaluator", "unknown"), "full_assignment": bool(full_assignment),
    }
    (OUTPUT_DIR / "tndp_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "tndp_report.md").write_text("\n".join([
        "# TNDP — синтез маршрутной сети", "",
        f"- Транспортных зон: **{report['n_zones']:,}**",
        f"- Остановок ОТ: **{report['n_stops']:,}**",
        f"- Коридоров OD: **{report['n_corridors']:,}**",
        f"- Кандидатных маршрутов: **{report['n_candidates']:,}**",
        f"- Итоговых маршрутов: **{report['n_routes']:,}**",
        f"- Доля обслуженного спроса: **{report['direct_demand_share'] * 100:.1f}%**",
        f"- Средние пересадки: **{report['transfers']:.2f}**",
        f"- Пользовательская стоимость: **{report['user_cost']:.2f}**",
        f"- Суммарная длина: **{report['operator_route_km']:.1f} км**",
        f"- Оценщик: **{report['evaluator']}**", "",
        "OD задаётся между полигонами транспортных зон. Маршруты привязаны к реальным остановкам и экспортируются вдоль дорожного графа.",
    ]), encoding="utf-8")
    return report
