"""End-to-end TNDP route synthesis from the Tranmodel OD matrix."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np

from config import CACHE_DIR, LAYERS_DIR, REPORT_DIR
from src.aequilibrae_pipeline import _open_project, build_project
from .aequilibrae_eval import AequilibraEEvaluationError, evaluate_route_set_aequilibrae
from .candidates import generate_route_candidates
from .corridors import extract_demand_corridors
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
    stop_proj = stops.to_crs("EPSG:4326")
    stop_xy = np.column_stack([
        stop_proj.geometry.x.to_numpy(dtype=float),
        stop_proj.geometry.y.to_numpy(dtype=float),
    ])
    terminal_nodes = set(np.flatnonzero(stops["is_terminal"].fillna(False).to_numpy()).tolist())
    return demand, stops, stop_graph, stop_xy, terminal_nodes


def _evaluate_with_aequilibrae(
    route_set: RouteSet,
    demand: np.ndarray,
    stop_xy: np.ndarray,
    project_path,
    config: NetworkDesignConfig,
) -> Evaluation:
    if route_set.route_count() == 0:
        return Evaluation(
            score=float(demand.sum() * config.uncovered_demand_weight),
            uncovered_demand=float(demand.sum()),
            direct_demand_share=0.0,
            metadata={"evaluator": "empty-network baseline"},
        )
    return evaluate_route_set_aequilibrae(
        route_set,
        demand,
        stop_xy,
        project_path,
        config,
        cache_dir=EVAL_CACHE,
    )


def run_tndp(
    config: NetworkDesignConfig | None = None,
    *,
    full_assignment: bool = True,
) -> dict:
    """Generate a route network from OD demand.

    Candidate screening is performed with the fast surrogate. The best
    singleton candidates are then evaluated with the actual AequilibraE
    TransitAssignment/Optimal Strategies procedure. The optimizer operates on
    that full evaluator, so every accepted network change is validated by a
    public-transport assignment.
    """
    config = config or NetworkDesignConfig()
    demand, stops, graph, stop_xy, terminal_nodes = _stop_graph_and_inputs()
    corridors = extract_demand_corridors(
        demand,
        stop_xy,
        top_pairs=config.corridor_top_pairs,
        max_distance_km=config.corridor_distance_km,
    )
    demand_vector = demand.sum(axis=1) + demand.sum(axis=0)
    candidates = generate_route_candidates(
        corridors,
        graph,
        stop_xy,
        node_ids=list(range(len(stops))),
        demand_vector=demand_vector,
        terminal_nodes=terminal_nodes,
        config=config,
    )
    if not candidates:
        raise RuntimeError("TNDP generated no feasible route candidates")

    # Avoid running a full TransitAssignment for hundreds of weak candidates.
    # Score one-route sets cheaply, then keep a controlled shortlist.
    singleton_scores = []
    for route in candidates:
        ev = surrogate_evaluator(demand, stop_xy, RouteSet([route]), config)
        singleton_scores.append((ev.score, route))
    singleton_scores.sort(key=lambda x: x[0])
    shortlist_n = min(
        len(singleton_scores),
        max(config.candidate_limit_per_corridor * 4, config.min_routes * 3, 24),
    )
    shortlist = [route for _, route in singleton_scores[:shortlist_n]]

    if full_assignment:
        project_path = build_project(force=False)

        def evaluator(route_set: RouteSet):
            try:
                return _evaluate_with_aequilibrae(
                    route_set, demand, stop_xy, project_path, config
                )
            except AequilibraEEvaluationError:
                raise
    else:
        project_path = None
        evaluator = lambda route_set: surrogate_evaluator(demand, stop_xy, route_set, config)

    optimizer = TNDPOptimizer(shortlist, evaluator, config)
    result = optimizer.solve()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    route_path = save_route_set(result.routes, OUTPUT_DIR / "generated_routes.json")
    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "backend": "Tranmodel TNDP solver",
        "n_stops": int(len(stops)),
        "n_terminals": int(len(terminal_nodes)),
        "n_corridors": int(len(corridors)),
        "n_candidates": int(len(candidates)),
        "n_screened_candidates": int(len(shortlist)),
        "n_routes": int(result.routes.route_count()),
        "score": float(result.evaluation.score),
        "user_cost": float(result.evaluation.user_cost),
        "direct_demand_share": float(result.evaluation.direct_demand_share),
        "uncovered_demand": float(result.evaluation.uncovered_demand),
        "transfers": float(result.evaluation.transfers),
        "operator_route_km": float(result.evaluation.operator_cost),
        "route_set": str(route_path),
        "evaluator": result.evaluation.metadata.get("evaluator", "unknown"),
        "full_assignment": bool(full_assignment),
    }
    (OUTPUT_DIR / "tndp_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "tndp_report.md").write_text(
        "\n".join([
            "# TNDP — синтез маршрутной сети",
            "",
            f"- Остановок-кандидатов: **{report['n_stops']:,}**",
            f"- Терминальных остановок: **{report['n_terminals']:,}**",
            f"- OD-коридоров: **{report['n_corridors']:,}**",
            f"- Кандидатных маршрутов: **{report['n_candidates']:,}**",
            f"- Кандидатов после быстрого отбора: **{report['n_screened_candidates']:,}**",
            f"- Итоговых маршрутов: **{report['n_routes']:,}**",
            f"- Доля обслуженного спроса: **{report['direct_demand_share'] * 100:.1f}%**",
            f"- Необслуженный спрос: **{report['uncovered_demand']:,.1f}**",
            f"- Средние пересадки: **{report['transfers']:.2f}**",
            f"- Пользовательская стоимость: **{report['user_cost']:.2f} мин/поездку**",
            f"- Суммарная длина маршрутов: **{report['operator_route_km']:.1f} км**",
            "",
            "## Оценивание",
            f"- Полное назначение AequilibraE: **{'да' if report['full_assignment'] else 'нет'}**",
            f"- Оценщик: **{report['evaluator']}**",
            "",
            "Принятые изменения маршрутной сети проверяются через общественно-транспортное назначение AequilibraE.",
        ]),
        encoding="utf-8",
    )
    return report
