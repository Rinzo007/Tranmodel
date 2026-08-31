from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .model import Evaluation, NetworkDesignConfig, RouteSet


class AequilibraEEvaluationError(RuntimeError):
    """Raised when AequilibraE cannot evaluate a route set."""


def _route_set_key(route_set: RouteSet) -> str:
    payload = [
        {
            "nodes": list(route.nodes),
            "frequency": float(route.frequency),
        }
        for route in route_set.routes
    ]
    return str(abs(hash(json.dumps(payload, sort_keys=True))))


def evaluate_route_set_aequilibrae(
    route_set: RouteSet,
    demand: np.ndarray,
    stop_xy_lonlat: np.ndarray,
    project_path: str | Path,
    config: NetworkDesignConfig,
    cache_dir: str | Path | None = None,
) -> Evaluation:
    """Evaluate a candidate route set using AequilibraE TransitAssignment."""
    total = float(np.asarray(demand, dtype=float).sum())
    if not route_set.routes:
        return Evaluation(
            score=total * config.uncovered_demand_weight,
            uncovered_demand=total,
            direct_demand_share=0.0,
            metadata={"evaluator": "AequilibraE", "empty_network": True},
        )

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

        gtfs = build_gtfs_from_route_set(route_set, stop_xy_lonlat, temp_root / "routes.zip")
        project = Project.from_path(project_dir)
        transit = Transit(project)
        builder = transit.new_gtfs_builder(
            agency="TRANMODEL",
            file_path=str(gtfs),
            day="2026-01-15",
            description="TNDP candidate route set",
        )
        # In current AequilibraE, GTFS loading/saving belongs to the
        # GTFSRouteSystemBuilder. Older versions exposed save_to_disk() on
        # Transit itself, but that method is not present in newer releases.
        builder.load_date("2026-01-15")
        if hasattr(builder, "set_allow_map_match"):
            builder.set_allow_map_match(False)
        builder.save_to_disk()

        graph_builder = transit.create_graph(
            with_inner_stop_transfers=True,
            with_outer_stop_transfers=False,
            with_walking_edges=True,
            distance_upper_bound=800.0,
            blocking_centroid_flows=True,
            connector_method="nearest_neighbour",
            max_connectors_per_zone=3,
        )
        try:
            graph_builder.create_line_geometry(method="connector project match", graph="c")
        except Exception:
            graph_builder.create_line_geometry(method="direct")
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
        assignment.set_algorithm("os")
        assignment.set_skimming_fields([
            "trav_time",
            "on_board_trav_time",
            "walking_trav_time",
            "waiting_time",
            "transfer_time",
            "boardings",
            "transfers",
        ])
        transit_class.set_demand_matrix_core("pt")
        assignment.execute()

        skim = assignment.get_skim_results()["pt"].matrix
        generalized = np.zeros_like(demand, dtype=float)
        for field, weight in (
            ("trav_time", config.in_vehicle_weight),
            ("waiting_time", config.wait_weight),
            ("walking_trav_time", config.walk_weight),
        ):
            if field in skim:
                generalized += np.nan_to_num(np.asarray(skim[field]), nan=np.inf) * weight
        if "transfers" in skim:
            generalized += (
                np.nan_to_num(np.asarray(skim["transfers"]), nan=np.inf)
                * config.transfer_penalty_min
                * config.transfer_weight
            )
        finite = np.isfinite(generalized)
        served = float(demand[finite].sum())
        uncovered = float(demand[~finite].sum())
        weighted_user_cost = float(
            np.nansum(demand[finite] * generalized[finite]) / max(served, 1.0)
        )
        transfer_arr = np.nan_to_num(
            np.asarray(skim.get("transfers", np.zeros_like(demand))), nan=0.0
        )
        result = Evaluation(
            score=weighted_user_cost + uncovered * config.uncovered_demand_weight,
            uncovered_demand=uncovered,
            direct_demand_share=float(served / max(total, 1.0)),
            metadata={
                "evaluator": "AequilibraE",
                "served_demand": served,
                "weighted_user_cost_min": weighted_user_cost,
                "average_transfers": float(
                    np.nansum(demand[finite] * transfer_arr[finite]) / max(served, 1.0)
                ),
                "routes": len(route_set.routes),
            },
        )
        result_path.write_text(json.dumps(result.__dict__, ensure_ascii=False), encoding="utf-8")
        return result
    finally:
        if project is not None:
            try:
                project.close()
            except Exception:
                pass
        shutil.rmtree(temp_root, ignore_errors=True)


def build_gtfs_from_route_set(route_set: RouteSet, stop_xy_lonlat: np.ndarray, output_zip: Path) -> Path:
    """Build a minimal GTFS feed for a candidate route set."""
    import csv
    import zipfile

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[dict[str, Any]]] = {
        "agency.txt": [{"agency_id": "TRANMODEL", "agency_name": "Tranmodel", "agency_url": "https://example.com", "agency_timezone": "Europe/Moscow"}],
        "routes.txt": [],
        "stops.txt": [],
        "trips.txt": [],
        "stop_times.txt": [],
        "calendar.txt": [{"service_id": "WKD", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1, "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20260101", "end_date": "20261231"}],
    }

    used_stops = sorted({int(node) for route in route_set.routes for node in route.nodes})
    for node in used_stops:
        lon, lat = map(float, stop_xy_lonlat[node])
        files["stops.txt"].append({"stop_id": str(node), "stop_name": f"Stop {node}", "stop_lat": lat, "stop_lon": lon})

    for ridx, route in enumerate(route_set.routes, start=1):
        route_id = f"R{ridx}"
        files["routes.txt"].append({"route_id": route_id, "agency_id": "TRANMODEL", "route_short_name": route_id, "route_long_name": route_id, "route_type": 3})
        files["trips.txt"].append({"route_id": route_id, "service_id": "WKD", "trip_id": f"T{ridx}"})
        for seq, node in enumerate(route.nodes):
            files["stop_times.txt"].append({"trip_id": f"T{ridx}", "arrival_time": f"08:{seq:02d}:00", "departure_time": f"08:{seq:02d}:30", "stop_id": str(node), "stop_sequence": seq + 1})

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, rows in files.items():
            if not rows:
                continue
            columns = list(rows[0].keys())
            lines: list[str] = []
            import io
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            zf.writestr(name, buffer.getvalue())
    return output_zip
