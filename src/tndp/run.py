"""End-to-end TNDP route synthesis from the Tranmodel OD matrix."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np

from config import CACHE_DIR, LAYERS_DIR, REPORT_DIR
from src.aequilibrae_pipeline import build_project
from .aequilibrae_eval import AequilibraEEvaluationError, evaluate_route_set_aequilibrae
from .candidates import generate_route_candidates
from .corridors import extract_demand_corridors
from .export import routes_to_geojson
from .io import load_phase2_demand, save_route_set
from .model import Evaluation, NetworkDesignConfig, RouteSet
from .network import add_stop_nodes, build_tndp_graph, snap_stops_to_graph
from .optimizer import TNDPOptimizer, surrogate_evaluator

OUTPUT_DIR = CACHE_DIR / "tndp"
EVAL_CACHE = OUTPUT_DIR / "evaluation_cache"


def _stop_graph_and_inputs():
    demand, stops = load_phase2_demand()
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    road_graph = build_tndp_graph(roads)
    _, stop_mapping, _ = snap_stops_to_graph(road_graph, stops)
    stop_graph = add_stop_nodes(road_graph, stop_mapping, k_neighbors=8)

    projected = stops.to_crs("EPSG:32637").reset_index(drop=True)
    stop_xy_km = np.column_stack([
        projected.geometry.x.to_numpy(dtype=float) / 1000.0,
        projected.geometry.y.to_numpy(dtype=float) / 1000.0,
    ])
    lonlat = stops.to_crs("EPSG:4326").reset_index(drop=True)
    stop_xy_lonlat = np.column_stack([
        lonlat.geometry.x.to_numpy(dtype=float),
        lonlat.geometry.y.to_numpy(dtype=float),
    ])
    terminal_nodes = set(np.flatnonzero(
        stops["is_terminal"].fillna(False).to_numpy()
    ).tolist())
    return demand, stops, stop_graph, stop_xy_km, stop_xy_lonlat, terminal_nodes


def _empty_evaluation(demand: np.ndarray, config: NetworkDesignConfig) -> Evaluation:
    total = float(demand.sum())
    return Evaluation(
        score=total * config.uncovered_demand_weight,
        uncovered_demand=total,
        direct_demand_share=0.0,
        metadata={"evaluator": "empty-network baseline"},
    )


def run_tndp(config: NetworkDesignConfig | None = None, *, full_assignment: bool = True) -> dict:
    """Generate and optimize a public-transport network from the OD matrix."""
    config = config or NetworkDesignConfig()
    config.validate()
    demand, stops, graph, stop_xy_km, stop_xy_lonlat, terminal_nodes = _stop_graph_and_inputs()

    corridors = extract_demand_corridors(
        demand, stop_xy_km, top_pairs=config.corridor_top_pairs,
        max_distance_km=config.corridor_distance_km,
    )
    demand_vector = demand.sum(axis=1) + demand.sum(axis=0)
    candidates = generate_route_candidates(
        corridors, graph, stop_xy_km,
        node_ids=list(range(len(stops))),
        demand_vector=demand_vector,
        terminal_nodes=terminal_nodes,
        config=config,
    )
    if not candidates:
        raise RuntimeError("TNDP generated no feasible route candidates")

    singleton = sorted(
        ((surrogate_evaluator(demand, stop_xy_km, RouteSet([r]), config).score, r)
         for r in candidates),
        key=lambda x: x[0],
    )
    shortlist_n = min(
        len(singleton),
        max(config.min_routes * 3, config.candidate_limit_per_corridor * 4, 24),
    )
    shortlist = [r for _, r in singleton[:shortlist_n]]

    if full_assignment:
        project_path = build_project(force=False)

        def evaluator(route_set: RouteSet):
            if not route_set.route_count():
                return _empty_evaluation(demand, config)
            return evaluate_route_set_aequilibrae(
                route_set, demand, stop_xy_lonlat, project_path, config,
                cache_dir=EVAL_CACHE,
            )
    else:
        evaluator = lambda route_set: surrogate_evaluator(demand, stop_xy_km, route_set, config)

    optimizer = TNDPOptimizer(shortlist, evaluator, config)
    result = optimizer.solve(graph=graph)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    route_path = save_route_set(result.routes, OUTPUT_DIR / "generated_routes.json")
    geojson_path = routes_to_geojson(result.routes, stops, OUTPUT_DIR / "generated_routes.geojson")
    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ev = result.evaluation
    report = {
        "backend": "Tranmodel TNDP solver",
        "n_stops": int(len(stops)),
        "n_terminals": int(len(terminal_nodes)),
        "n_corridors": int(len(corridors)),
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
    }
    (OUTPUT_DIR / "tndp_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "tndp_report.md").write_text("\n".join([
        "# TNDP — синтез маршрутной сети", "",
        f"- Остановок-кандидатов: **{report['n_stops']:,}**",
        f"- Терминальных остановок: **{report['n_terminals']:,}**",
        f"- OD-коридоров: **{report['n_corridors']:,}**",
        f"- Кандидатных маршрутов: **{report['n_candidates']:,}**",
        f"- После предварительного отбора: **{report['n_screened_candidates']:,}**",
        f"- Итоговых маршрутов: **{report['n_routes']:,}**",
        f"- Доля обслуженного спроса: **{report['direct_demand_share'] * 100:.1f}%**",
        f"- Необслуженный спрос: **{report['uncovered_demand']:,.1f}**",
        f"- Средние пересадки: **{report['transfers']:.2f}**",
        f"- Пользовательская стоимость: **{report['user_cost']:.2f}**",
        f"- Суммарная длина маршрутов: **{report['operator_route_km']:.1f} км**",
        f"- Оценщик: **{report['evaluator']}**",
        f"- Геометрия: `{report['route_geojson']}`",
        "",
        "Поиск выполняется целыми маршрутами; локальные операции: remove, extend, shorten, replace.",
    ]), encoding="utf-8")
    return report
