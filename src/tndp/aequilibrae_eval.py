from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from config import PROJ_EPSG
from .gtfs import build_gtfs_from_route_set
from .model import Evaluation, NetworkDesignConfig, RouteSet

EVALUATOR_VERSION = "aeq-transit-v5-gtfs-import"


class AequilibraEEvaluationError(RuntimeError):
    """Raised when AequilibraE cannot evaluate a route set."""


def _route_frequency(route: Any) -> float:
    return float(getattr(route, "frequency_vph", getattr(route, "frequency", 6.0)))


def _route_length_km(route: Any, road_graph: nx.Graph, stop_mapping) -> float:
    total = 0.0
    for a, b in zip(route.nodes[:-1], route.nodes[1:]):
        path = nx.shortest_path(road_graph, stop_mapping[int(a)], stop_mapping[int(b)], weight="time")
        total += float(nx.path_weight(road_graph, path, weight="length_km"))
    return total


def _route_set_key(route_set: RouteSet) -> str:
    payload = {"version": EVALUATOR_VERSION, "routes": [
        {"nodes": list(route.nodes), "frequency_vph": _route_frequency(route)}
        for route in route_set.routes
    ]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _evaluation_json(value: Evaluation) -> str:
    return json.dumps(asdict(value), ensure_ascii=False, default=float, sort_keys=True)


def evaluate_route_set_aequilibrae(route_set: RouteSet, demand: np.ndarray, stop_xy_lonlat: np.ndarray,
                                   project_path: str | Path, config: NetworkDesignConfig, *,
                                   road_graph: nx.Graph, stop_mapping, cache_dir: str | Path | None = None) -> Evaluation:
    total = float(np.asarray(demand, dtype=float).sum())
    if not route_set.routes:
        return Evaluation(score=total * config.uncovered_demand_weight, uncovered_demand=total,
                          direct_demand_share=0.0, metadata={"evaluator": "AequilibraE", "empty_network": True})

    key = _route_set_key(route_set)
    root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="tranmodel_tndp_eval_"))
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / f"{key}.json"
    if result_path.exists():
        return Evaluation(**json.loads(result_path.read_text(encoding="utf-8")))

    try:
        from aequilibrae import Project
        from aequilibrae.matrix import AequilibraeMatrix
        from aequilibrae.paths import TransitAssignment, TransitClass
        from aequilibrae.transit import Transit
    except ImportError as exc:
        raise AequilibraEEvaluationError("AequilibraE is required for full TNDP evaluation") from exc

    temp_root = Path(tempfile.mkdtemp(prefix="tranmodel_tndp_project_"))
    project = None
    try:
        project_dir = temp_root / "project"
        shutil.copytree(Path(project_path), project_dir, dirs_exist_ok=True)
        public_db = project_dir / "public_transport.sqlite"
        if public_db.exists():
            public_db.unlink()

        gtfs = build_gtfs_from_route_set(route_set, stop_xy_lonlat, temp_root / "routes.zip",
                                         road_graph=road_graph, stop_mapping=stop_mapping)
        project = Project.from_path(project_dir)
        transit = Transit(project)

        # IMPORTANT: the builder's `day` must start empty. If it is initialized
        # with the same date passed to load_date(), AequilibraE treats the date
        # as already loaded and skips GTFS parsing, leaving all transit tables empty.
        builder = transit.new_gtfs_builder(
            agency="TRANMODEL", file_path=str(gtfs), day="",
            description="TNDP candidate route set"
        )
        builder.set_allow_map_match(False)
        builder.load_date("2026-01-15")
        # load_date() prepares the in-memory route system; execute_import()
        # persists it into public_transport.sqlite. Calling save_to_disk()
        # directly is not sufficient when the builder has not executed import.
        builder.execute_import()

        graph_builder = transit.create_graph(
            projected_crs=f"EPSG:{PROJ_EPSG}",
            with_inner_stop_transfers=True,
            with_outer_stop_transfers=False,
            with_walking_edges=True,
            distance_upper_bound=800.0,
            blocking_centroid_flows=True,
            connector_method="nearest_neighbour",
            max_connectors_per_zone=3,
        )
        graph_builder.create_line_geometry(method="connector project match", graph="c")
        graph_builder.save()
        graph = graph_builder.to_transit_graph()
        centroids = np.asarray(graph.centroids, dtype=np.int64)
        if len(centroids) != demand.shape[0]:
            raise AequilibraEEvaluationError(
                f"Transit graph has {len(centroids)} zone centroids but demand has {demand.shape[0]} zones"
            )

        matrix = AequilibraeMatrix()
        matrix.create_empty(zones=len(centroids), matrix_names=["pt"], memory_only=True)
        matrix.index = centroids
        matrix.matrices[:, :, 0] = demand
        matrix.computational_view(["pt"])
        transit_class = TransitClass(name="pt", graph=graph, matrix=matrix)
        assignment = TransitAssignment()
        assignment.add_class(transit_class)
        assignment.set_time_field("trav_time")
        assignment.set_frequency_field("freq")
        assignment.set_skimming_fields([
            "trav_time", "on_board_trav_time", "walking_trav_time",
            "waiting_time", "transfer_time", "boardings", "transfers",
        ])
        assignment.set_algorithm("os")
        transit_class.set_demand_matrix_core("pt")
        assignment.execute()

        skim = assignment.get_skim_results()["pt"].matrix
        generalized = np.zeros_like(demand, dtype=float)
        for field, weight in (("trav_time", config.in_vehicle_weight),
                              ("waiting_time", config.wait_weight),
                              ("walking_trav_time", config.walk_weight)):
            if field in skim:
                generalized += np.nan_to_num(np.asarray(skim[field]), nan=np.inf) * weight
        if "transfers" in skim:
            generalized += np.nan_to_num(np.asarray(skim["transfers"]), nan=np.inf) * config.transfer_penalty_min * config.transfer_weight
        finite = np.isfinite(generalized)
        served = float(demand[finite].sum())
        uncovered = float(demand[~finite].sum())
        weighted_user_cost = float(np.nansum(demand[finite] * generalized[finite]) / max(served, 1.0))
        transfer_arr = np.nan_to_num(np.asarray(skim.get("transfers", np.zeros_like(demand))), nan=0.0)
        avg_transfers = float(np.nansum(demand * transfer_arr) / max(total, 1.0))
        direct_share = float(demand[(finite) & (transfer_arr == 0)].sum() / max(total, 1.0))
        operator_km = sum(_route_length_km(route, road_graph, stop_mapping) * _route_frequency(route) for route in route_set.routes)
        score = (weighted_user_cost + operator_km * config.operator_route_km_weight
                 + uncovered * config.uncovered_demand_weight
                 + avg_transfers * config.transfer_penalty_min * config.transfer_weight)
        evaluation = Evaluation(score=float(score), user_cost=weighted_user_cost,
                                operator_cost=float(operator_km), uncovered_demand=uncovered,
                                transfers=avg_transfers, direct_demand_share=direct_share,
                                metadata={"evaluator": "AequilibraE", "served_demand": served})
        result_path.write_text(_evaluation_json(evaluation), encoding="utf-8")
        return evaluation
    finally:
        if project is not None:
            try:
                project.close()
            except Exception:
                pass
        shutil.rmtree(temp_root, ignore_errors=True)
