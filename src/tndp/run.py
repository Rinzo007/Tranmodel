"""End-to-end route synthesis from the current Tranmodel OD matrix."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np

from config import CACHE_DIR, LAYERS_DIR, REPORT_DIR
from .candidates import generate_route_candidates
from .corridors import extract_demand_corridors
from .io import load_phase2_demand, save_route_set
from .model import NetworkDesignConfig, RouteSet
from .network import add_stop_nodes, build_tndp_graph, snap_stops_to_graph
from .optimizer import TNDPOptimizer, surrogate_evaluator

OUTPUT_DIR = CACHE_DIR / "tndp"


def _stop_graph_and_inputs():
    demand, stops = load_phase2_demand()
    roads = gpd.read_parquet(LAYERS_DIR / "roads.parquet")
    road_graph = build_tndp_graph(roads)
    _, stop_mapping, _ = snap_stops_to_graph(road_graph, stops)
    stop_graph = add_stop_nodes(road_graph, stop_mapping, k_neighbors=8)
    stop_proj = stops.to_crs("EPSG:32637")
    stop_xy = np.column_stack([
        stop_proj.geometry.x.to_numpy() / 1000.0,
        stop_proj.geometry.y.to_numpy() / 1000.0,
    ])
    terminal_nodes = set(np.flatnonzero(stops["is_terminal"].fillna(False).to_numpy()).tolist())
    return demand, stops, stop_graph, stop_xy, terminal_nodes


def run_tndp(config: NetworkDesignConfig | None = None) -> dict:
    """Generate a route network from the current OD matrix.

    Candidate generation uses terminal restrictions and network shortest paths.
    The current evaluator is a fast whole-route surrogate; it is intentionally
    separated from the solver so AequilibraE Transit/Optimal Strategies can be
    substituted without changing candidate generation or optimization.
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

    def evaluator(route_set: RouteSet):
        return surrogate_evaluator(demand, stop_xy, route_set, config)

    optimizer = TNDPOptimizer(candidates, evaluator, config)
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
        "n_routes": int(result.routes.route_count()),
        "score": float(result.evaluation.score),
        "direct_demand_share": float(result.evaluation.direct_demand_share),
        "uncovered_demand": float(result.evaluation.uncovered_demand),
        "operator_route_km": float(result.evaluation.operator_cost),
        "route_set": str(route_path),
        "evaluator": "whole-route surrogate",
        "next_evaluator": "AequilibraE Transit / Optimal Strategies",
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
            f"- Итоговых маршрутов: **{report['n_routes']:,}**",
            f"- Прямо обслуживаемая доля спроса (суррогатная оценка): **{report['direct_demand_share'] * 100:.1f}%**",
            f"- Необслуженный спрос: **{report['uncovered_demand']:,.1f}**",
            f"- Суммарная длина маршрутов: **{report['operator_route_km']:.1f} км**",
            "",
            "## Статус оценивания",
            "Синтез маршрутов и ограничения уже отделены от оценщика. Текущий оценщик быстрый и предназначен для отбора кандидатов; для итоговой оптимизации его необходимо подключить к AequilibraE Transit Assignment.",
        ]),
        encoding="utf-8",
    )
    return report
