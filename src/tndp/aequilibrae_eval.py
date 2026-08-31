"""Full AequilibraE evaluator for TNDP route sets.

A candidate RouteSet is converted to a temporary GTFS feed, imported into a
throw-away copy of the AequilibraE project, assigned with Optimal Strategies,
and scored using user and operator costs. This is intentionally separated from
candidate generation so a cheap surrogate can screen candidates before the
expensive full evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .model import Evaluation, NetworkDesignConfig, RouteSet


class AequilibraEEvaluationError(RuntimeError):
    """Raised when a route-set cannot be evaluated by AequilibraE."""


def _hhmmss(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    h, rem = divmod(value, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    x1, y1 = a
    x2, y2 = b
    lat = math.radians((y1 + y2) / 2.0)
    dx = math.radians(x2 - x1) * 6371000.0 * math.cos(lat)
    dy = math.radians(y2 - y1) * 6371000.0
    return float(math.hypot(dx, dy))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AequilibraEEvaluationError(f"Cannot create empty GTFS table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_gtfs_from_route_set(
    route_set: RouteSet,
    stop_xy_lonlat: np.ndarray,
    output_zip: str | Path,
    *,
    headway_min: float = 10.0,
    service_start: int = 6 * 3600,
    service_end: int = 23 * 3600,
    speed_kmh: float = 22.0,
    dwell_sec: float = 20.0,
) -> Path:
    """Build a deterministic GTFS feed from a RouteSet.

    Route node IDs are integer stop indices into ``stop_xy_lonlat``. Each route
    gets two directions and uniform headway because TNDP candidates do not yet
    carry timetable data.
    """
    if route_set.route_count() == 0:
        raise AequilibraEEvaluationError("Cannot evaluate an empty route set")
    if speed_kmh <= 0 or headway_min <= 0:
        raise ValueError("speed_kmh and headway_min must be positive")

    target = Path(output_zip)
    work = Path(tempfile.mkdtemp(prefix="tranmodel_gtfs_"))
    try:
        stops = sorted({int(n) for r in route_set.routes for n in r.nodes})
        stop_rows = []
        for idx in stops:
            lon, lat = map(float, stop_xy_lonlat[idx])
            stop_rows.append({
                "stop_id": f"s{idx + 1}",
                "stop_name": f"Stop {idx + 1}",
                "stop_lat": lat,
                "stop_lon": lon,
                "location_type": 0,
            })

        agency = [{
            "agency_id": "TRANMODEL",
            "agency_name": "Tranmodel TNDP",
            "agency_url": "https://github.com/Rinzo007/Tranmodel",
            "agency_timezone": "Europe/Moscow",
            "agency_lang": "ru",
        }]
        calendar = [{
            "service_id": "daily",
            "monday": 1,
            "tuesday": 1,
            "wednesday": 1,
            "thursday": 1,
            "friday": 1,
            "saturday": 1,
            "sunday": 1,
            "start_date": "20260101",
            "end_date": "20261231",
        }]
        route_rows = []
        trip_rows = []
        stop_time_rows = []
        shape_rows = []
        departures = np.arange(service_start, service_end + 1, int(headway_min * 60))

        for route_num, route in enumerate(route_set.routes, start=1):
            route_id = f"r{route_num}"
            route_rows.append({
                "route_id": route_id,
                "agency_id": "TRANMODEL",
                "route_short_name": str(route.route_id or route_num),
                "route_long_name": f"TNDP route {route_num}",
                "route_type": 3,
            })
            seq_variants = (tuple(route.nodes), tuple(reversed(route.nodes)))
            for direction_id, seq in enumerate(seq_variants):
                shape_id = f"sh{route_num}_{direction_id}"
                for stop_seq, node in enumerate(seq, start=1):
                    lon, lat = map(float, stop_xy_lonlat[int(node)])
                    shape_rows.append({
                        "shape_id": shape_id,
                        "shape_pt_lat": lat,
                        "shape_pt_lon": lon,
                        "shape_pt_sequence": stop_seq,
                    })

                for trip_no, departure in enumerate(departures, start=1):
                    trip_id = f"t{route_num}_{direction_id}_{trip_no}"
                    trip_rows.append({
                        "route_id": route_id,
                        "service_id": "daily",
                        "trip_id": trip_id,
                        "trip_headsign": f"direction {direction_id + 1}",
                        "direction_id": direction_id,
                        "shape_id": shape_id,
                    })
                    elapsed = 0.0
                    for stop_seq, node in enumerate(seq, start=1):
                        if stop_seq > 1:
                            a = stop_xy_lonlat[int(seq[stop_seq - 2])]
                            b = stop_xy_lonlat[int(node)]
                            meters = _distance_m((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                            elapsed += meters / (speed_kmh * 1000.0 / 3600.0)
                            elapsed += dwell_sec
                        tm = _hhmmss(departure + elapsed)
                        stop_time_rows.append({
                            "trip_id": trip_id,
                            "arrival_time": tm,
                            "departure_time": tm,
                            "stop_id": f"s{int(node) + 1}",
                            "stop_sequence": stop_seq,
                        })

        files = {
            "agency.txt": agency,
            "stops.txt": stop_rows,
            "routes.txt": route_rows,
            "calendar.txt": calendar,
            "trips.txt": trip_rows,
            "stop_times.txt": stop_time_rows,
            "shapes.txt": shape_rows,
        }
        for name, rows in files.items():
            _write_csv(work / name, rows)

        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(work.glob("*.txt")):
                zf.write(path, path.name)
        return target
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _route_set_key(route_set: RouteSet) -> str:
    payload = [
        {"nodes": list(r.nodes), "freq": round(r.frequency_vph, 6)}
        for r in route_set.routes
    ]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluate_route_set_aequilibrae(
    route_set: RouteSet,
    demand_matrix: np.ndarray,
    stop_xy_lonlat: np.ndarray,
    project_path: str | Path,
    config: NetworkDesignConfig | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> Evaluation:
    """Evaluate a RouteSet using AequilibraE TransitAssignment/Optimal Strategies."""
    config = config or NetworkDesignConfig()
    demand_matrix = np.asarray(demand_matrix, dtype=float)
    if demand_matrix.ndim != 2 or demand_matrix.shape[0] != demand_matrix.shape[1]:
        raise ValueError("demand_matrix must be square")
    if len(stop_xy_lonlat) != demand_matrix.shape[0]:
        raise ValueError("stop_xy_lonlat and demand_matrix dimensions differ")

    key = _route_set_key(route_set)
    root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="tranmodel_tndp_eval_"))
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / f"{key}.json"
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return Evaluation(**data)

    try:
        from aequilibrae import Project
        from aequilibrae.matrix import AequilibraeMatrix
        from aequilibrae.paths import TransitAssignment, TransitClass
        from aequilibrae.transit import Transit
    except ImportError as exc:
        raise AequilibraEEvaluationError("AequilibraE is required for full TNDP evaluation") from exc

    project_src = Path(project_path)
    if not project_src.exists():
        raise FileNotFoundError(f"AequilibraE project not found: {project_src}")

    temp_project = Path(tempfile.mkdtemp(prefix="tranmodel_tndp_project_"))
    try:
        shutil.copytree(project_src, temp_project / "project", dirs_exist_ok=True)
        project_dir = temp_project / "project"
        public_db = project_dir / "public_transport.sqlite"
        if public_db.exists():
            public_db.unlink()

        gtfs = build_gtfs_from_route_set(route_set, stop_xy_lonlat, temp_project / "routes.zip")
        project = Project.from_path(project_dir)
        data = Transit(project)
        builder = data.new_gtfs_builder(
            agency="TRANMODEL",
            file_path=str(gtfs),
            day="2026-01-15",
            description="TNDP candidate route set",
        )
        builder.load_date("2026-01-15")
        data.save_to_disk()

        graph_builder = data.create_graph(
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

        mat = AequilibraeMatrix()
        mat.create_empty(zones=len(demand_matrix), matrix_names=["pt"], memory_only=True)
        mat.index = np.asarray(graph.centroids, dtype=np.int64)
        if len(mat.index) != demand_matrix.shape[0]:
            raise AequilibraEEvaluationError(
                f"Transit graph has {len(mat.index)} zones but demand has {demand_matrix.shape[0]}"
            )
        mat.matrices[:, :, 0] = demand_matrix
        mat.computational_view(["pt"])

        tc = TransitClass(name="pt", graph=graph, matrix=mat)
        assignment = TransitAssignment()
        assignment.add_class(tc)
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
        tc.set_demand_matrix_core("pt")
        assignment.execute()

        skim = assignment.get_skim_results()["pt"].matrix
        total = float(demand_matrix.sum())
        if total <= 0:
            raise AequilibraEEvaluationError("Demand matrix is empty")

        user_time = float(np.asarray(skim["trav_time"]).multiply if False else 0.0)
        # The matrix object exposes fields as numpy arrays; compute generalized
        # cost from available components while treating unavailable values as 0.
        generalized = np.zeros_like(demand_matrix, dtype=float)
        if "trav_time" in skim:
            generalized += np.nan_to_num(np.asarray(skim["trav_time"]), nan=0.0) * config.in_vehicle_weight
        if "waiting_time" in skim:
            generalized += np.nan_to_num(np.asarray(skim["waiting_time"]), nan=0.0) * config.wait_weight
        if "walking_trav_time" in skim:
            generalized += np.nan_to_num(np.asarray(skim["walking_trav_time"]), nan=0.0) * config.walk_weight
        if "transfers" in skim:
            generalized += np.nan_to_num(np.asarray(skim["transfers"]), nan=0.0) * config.transfer_penalty_min * config.transfer_weight

        finite = np.isfinite(generalized)
        weighted = float(np.nansum(demand_matrix[finite] * generalized[finite]) / total)
        transfer_arr = np.nan_to_num(np.asarray(skim.get("transfers", np.zeros_like(demand_matrix))), nan=0.0)
        avg_transfers = float(np.nansum(demand_matrix * transfer_arr) / total)

        load = tc.results.get_load_results()
        operator_km = 0.0
        for route in route_set.routes:
            for a, b in zip(route.nodes[:-1], route.nodes[1:]):
                pa = stop_xy_lonlat[int(a)]
                pb = stop_xy_lonlat[int(b)]
                operator_km += _distance_m((float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1]))) / 1000.0
        operator_cost = operator_km * config.operator_route_km_weight

        score = weighted + operator_cost + avg_transfers * config.transfer_weight
        result = Evaluation(
            score=float(score),
            user_cost=float(weighted),
            operator_cost=float(operator_km),
            uncovered_demand=0.0,
            transfers=float(avg_transfers),
            direct_demand_share=1.0,
            metadata={
                "evaluator": "AequilibraE TransitAssignment / Optimal Strategies",
                "assigned_link_rows": int(len(load)),
                "total_demand": total,
            },
        )
        result_path.write_text(json.dumps({
            "score": result.score,
            "user_cost": result.user_cost,
            "operator_cost": result.operator_cost,
            "uncovered_demand": result.uncovered_demand,
            "transfers": result.transfers,
            "direct_demand_share": result.direct_demand_share,
            "metadata": result.metadata,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        project.close()
        return result
    finally:
        shutil.rmtree(temp_project, ignore_errors=True)
