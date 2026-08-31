"""End-to-end TNDP route synthesis from zone-based OD demand."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from config import CACHE_DIR, LAYERS_DIR, REPORT_DIR
from src.aequilibrae_pipeline import build_project
from .aequilibrae_eval import evaluate_route_set_aequilibrae
from .candidates import generate_route_candidates
from .corridors import DemandCorridor, extract_demand_corridors
from .export import routes_to_geojson
from .frequency import required_frequency_vph
from .io import save_route_set
from .model import Evaluation, NetworkDesignConfig, RouteSet
from .network import add_stop_nodes, build_tndp_graph, snap_stops_to_graph
from .surrogate import surrogate_evaluator
from .zone_io import load_zone_demand

OUTPUT_DIR = CACHE_DIR / "tndp"
EVAL_CACHE = OUTPUT_DIR / "evaluation_cache"


def _load_inputs():
    demand, zones = load_zone_demand()
    stops = gpd.read_parquet(CACHE_DIR / "phase1_real" / "stops_demand.parquet").to_crs("EPSG:32637").reset_index(drop=True)
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    road_graph = build_tndp_graph(roads)
    _, stop_mapping, _ = snap_stops_to_graph(road_graph, stops)
    stop_graph = add_stop_nodes(road_graph, stop_mapping, k_neighbors=8)

    stop_xy = np.column_stack([stops.geometry.x.to_numpy(), stops.geometry.y.to_numpy()])
    zone_xy = np.column_stack([zones.geometry.centroid.x.to_numpy(), zones.geometry.centroid.y.to_numpy()])
    stop_tree = cKDTree(stop_xy)
    zone_to_stop = stop_tree.query(zone_xy, k=1)[1].astype(int)
    terminal_nodes = set(np.flatnonzero(stops["is_terminal"].fillna(False).to_numpy()).tolist())
    return demand, zones, stops, stop_graph, stop_xy, zone_xy, zone_to_stop, terminal_nodes


def _map_corridors_to_stops(corridors: list[DemandCorridor], zone_to_stop: np.ndarray) -> list[DemandCorridor]:
    """Convert zone-level corridor endpoints into transit-stop endpoints."""
    out: list[DemandCorridor] = []
    seen: set[tuple[int, int]] = set()
    for c in corridors:
        o = int(zone_to_stop[c.origin])
        d = int(zone_to_stop[c.destination])
        if o == d:
            continue
        key = (o, d)
        if key in seen:
            continue
        seen.add(key)
        out.append(DemandCorridor(o, d, c.demand, c.direct_distance_km))
    return out


def _empty_evaluation(demand: np.ndarray, config: NetworkDesignConfig) -> Evaluation:
    total = float(demand.sum())
    return Evaluation(score=total * config.uncovered_demand_weight, uncovered_demand=total,
                      direct_demand_share=0.0, metadata={"evaluator": "empty-network baseline"})


def run_tndp(config: NetworkDesignConfig | None = None, *, full_assignment: bool = True) -> dict:
    """Synthesize a transit route network from zone-based OD demand."""
    config = config or NetworkDesignConfig()
    config.validate()
    demand, zones, stops, graph, stop_xy, zone_xy, zone_to_stop, terminal_nodes = _load_inputs()

    zone_corridors = extract_demand_corridors(
        demand, zone_xy, top_pairs=config.corridor_top_pairs,
        max_distance_km=config.corridor_distance_km,
    )
    corridors = _map_corridors_to_stops(zone_corridors, zone_to_stop)
    stop_demand_vector = np.zeros(len(stops), dtype=float)
    production = zones.production.to_numpy(dtype=float) + zones.attraction.to_numpy(dtype=float)
    np.add.at(stop_demand_vector, zone_to_stop, production)

    candidates = generate_route_candidates(
        corridors, graph, stop_xy / 1000.0,
        node_ids=list(range(len(stops))), demand_vector=stop_demand_vector,
        terminal_nodes=terminal_nodes, config=config,
    )
    if not candidates:
        raise RuntimeError("TNDP generated no feasible route candidates")

    singleton = sorted(
        ((surrogate_evaluator(demand, zone_xy / 1000.0, RouteSet([r]), config).score, r)
         for r in candidates), key=lambda x: x[0]
    )
    shortlist = [r for _, r in singleton[:min(len(singleton), max(config.min_routes * 3, 24))]]

    if full_assignment:
        project_path = build_project(force=False)
        evaluator = lambda route_set: (
            _empty_evaluation(demand, config) if not route_set.route_count() else
            evaluate_route_set_aequilibrae(
                route_set, demand, stop_xy[:, ::-1] if False else _to_lonlat(stops),
                project_path, config, cache_dir=EVAL_CACHE,
            )
        )
    else:
        evaluator = lambda route_set: surrogate_evaluator(demand, zone_xy / 1000.0, route_set, config)

    optimizer = __import__("src.tndp.optimizer", fromlist=["TNDPOptimizer"]).TNDPOptimizer(shortlist, evaluator, config)
    result = optimizer.solve(graph=graph)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    route_path = save_route_set(result.routes, OUTPUT_DIR / "generated_routes.json")
    geojson_path = routes_to_geojson(result.routes, stops, OUTPUT_DIR / "generated_routes.geojson")

    ev = result.evaluation
    report = {
        "backend": "Tranmodel TNDP solver",
        "demand_units": "transport zones",
        "network_units": "road graph + transit stops",
        "n_zones": int(len(zones)),
        "n_stops": int(len(stops)),
        "n_terminals": int(len(terminal_nodes)),
        "n_corridors": int(len(zone_corridors)),
        "n_candidates": int(len(candidates)),
        "n_screened_candidates": int(len(shortlist)),
        "n_routes": int(result.routes.route_count()),
        "score": float(ev.score),
        "user_cost": float(ev.user_cost),
        "direct_demand_share": float(ev.direct_demand_share),
        "uncovered_demand": float(ev.uncovered_demand),
        "transfers": float(ev.transfers),
        "operator_route_km": float(ev.operator_cost),
        "capacity_excess": float(ev.capacity_excess),
        "route_set": str(route_path),
        "route_geojson": str(geojson_path),
        "evaluator": ev.metadata.get("evaluator", "unknown"),
        "full_assignment": bool(full_assignment),
        "zone_size_m": float((zones.geometry.area.median() ** 0.5)),
    }
    (OUTPUT_DIR / "history.json").write_text(json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "tndp_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "tndp_report.md").write_text("\n".join([
        "# TNDP — синтез маршрутной сети",
        "",
        f"- Транспортных зон: **{report['n_zones']:,}**",
        f"- Остановок ОТ: **{report['n_stops']:,}**",
        f"- Коридоров OD: **{report['n_corridors']:,}**",
        f"- Кандидатов маршрутов: **{report['n_candidates']:,}**",
        f"- Итоговых маршрутов: **{report['n_routes']:,}**",
        f"- Обслужено спроса: **{report['direct_demand_share'] * 100:.1f}%**",
        f"- Средние пересадки: **{report['transfers']:.2f}**",
        f"- Суммарная длина: **{report['operator_route_km']:.1f} км**",
        f"- Оценщик: **{report['evaluator']}**",
        "",
        "OD задаётся между полигонами транспортных зон; маршруты строятся по дорожному графу и проходят через реальные остановки ОТ.",
    ]), encoding="utf-8")
    return report


def _to_lonlat(stops: gpd.GeoDataFrame) -> np.ndarray:
    wgs = stops.to_crs("EPSG:4326")
    return np.column_stack([wgs.geometry.x.to_numpy(), wgs.geometry.y.to_numpy()])
