"""Full AequilibraE evaluator for TNDP route sets."""

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
    service_date: str = "2026-01-15",
    default_headway_min: float = 10.0,
    service_start: int = 6 * 3600,
    service_end: int = 23 * 3600,
    speed_kmh: float = 22.0,
    dwell_sec: float = 20.0,
) -> Path:
    """Build a deterministic GTFS feed from a RouteSet.

    Route nodes are integer indices into stop_xy_lonlat. Each route is emitted
    in both directions. Route.frequency_vph controls departures; the fallback
    headway is used only when frequency is invalid.
    """
    if route_set.route_count() == 0:
        raise AequilibraEEvaluationError("Cannot evaluate an empty route set")
    if speed_kmh <= 0 or default_headway_min <= 0:
        raise ValueError("speed_kmh and default_headway_min must be positive")

    target = Path(output_zip)
    work = Path(tempfile.mkdtemp(prefix="tranmodel_gtfs_"))
    try:
        used = sorted({int(n) for route in route_set.routes for n in route.nodes})
        stop_rows = []
        for idx in used:
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
            "start_date": service_date.replace("-", ""),
            "end_date": "20261231",
        }]
        route_rows, trip_rows, stop_time_rows, shape_rows = [], [], [], []

        for route_num, route in enumerate(route_set.routes, start=1):
            route_id = f"r{route_num}"
            route_rows.append({
                "route_id": route_id,
                "agency_id": "TRANMODEL",
                "route_short_name": str(route.route_id or route_num),
                "route_long_name": f"TNDP route {route_num}",
                "route_type": 3,
            })
            frequency = float(route.frequency_vph)
            if frequency <= 0:
                frequency = 60.0 / default_headway_min
            headway_sec = max(60, int(round(3600.0 / frequency)))
            departures = np.arange(service_start, service_end + 1, headway_sec)

            for direction_id, seq in enumerate((tuple(route.nodes), tuple(reversed(route.nodes)))):
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

        for name, rows in {
            "agency.txt": agency,
            "stops.txt": stop_rows,
            "routes.txt": route_rows,
            "calendar.txt": calendar,
            "trips.txt": trip_rows,
            "stop_times.txt": stop_time_rows,
            "shapes.txt": shape_rows,
        }.items():
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
    demand = np.asarray(demand_matrix, dtype=float)
    if demand.ndim != 2 or demand.shape[0] != demand.shape[1]:
        raise ValueError("demand_matrix must be square")
    if len(stop_xy_lonlat) != demand.shape[0]:
        raise ValueError("stop_xy_lonlat and demand_matrix dimensions differ")

    key = _route_set_key(route_set)
    owned_cache = cache_dir is None
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

    source = Path(project_path)
    if not source.exists():
        raise FileNotFoundError(f"AequilibraE project not found: {source}")

    temp_root = Path(tempfile.mkdtemp(prefix="tranmodel_tndp_project_"))
    project = None
    try:
        project_dir = temp_root / "project"
        shutil.copytree(source, project_dir, dirs_exist_ok=True)
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
        builder.load_date("2026-01-15")
        transit.save_to_disk()

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
                f"Transit graph has {len(centroids)} centroids but demand has {demand.shape[0]} zones"
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
        total = float(demand.sum())
        if total <= 0:
            raise AequilibraEEvaluationError("Demand matrix is empty")

        generalized = np.zeros_like(demand, dtype=float)
        if "trav_time" in skim:
            generalized += np.nan_to_num(np.asarray(skim["trav_time"]), nan=np.inf) * config.in_vehicle_weight
        if "waiting_time" in skim:
            generalized += np.nan_to_num(np.asarray(skim["waiting_time"]), nan=np.inf) * config.wait_weight
        if "walking_trav_time" in skim:
            generalized += np.nan_to_num(np.asarray(skim["walking_trav_time"]), nan=np.inf) * config.walk_weight
        if "transfers" in skim:
            generalized += np.nan_to_num(np.asarray(skim["transfers"]), nan=np.inf) * config.transfer_penalty_min * config.transfer_weight

        finite = np.isfinite(generalized)
        served = float(demand[finite].sum())
        uncovered = float(demand[~finite].sum())
        denominator = max(served, 1.0)
        weighted_user_cost = float(np.nansum(demand[finite] * generalized[finite]) / denominator)

        transfers = np.nan_to_num(np.asarray(skim.get("transfers", np.zeros_like(demand))), nan=0.0)
        avg_transfers = float(np.nansum(demand * transfers) / total)

        operator_km = 0.0
        for route in route_set.routes:
            operator_km += sum(
                _distance_m((float(stop_xy_lonlat[int(a), 0]), float(stop_xy_lonlat[int(a), 1])),
                            (float(stop_xy_lonlat[int(b), 0]), float(stop_xy_lonlat[int(b), 1]))) / 1000.0
                for a, b in zip(route.nodes[:-1], route.nodes[1:])
            )

        score = (
            weighted_user_cost
            + avg_transfers * config.transfer_penalty_min * config.transfer_weight
            + uncovered * config.uncovered_demand_weight
            + operator_km * config.operator_route_km_weight
        )
        result = Evaluation(
            score=float(score),
            user_cost=float(weighted_user_cost),
            operator_cost=float(operator_km),
            uncovered_demand=uncovered,
            transfers=float(avg_transfers),
            direct_demand_share=float(served / total),
            metadata={
                "evaluator": "AequilibraE TransitAssignment / Optimal Strategies",
                "assigned_link_rows": int(len(transit_class.results.get_load_results())),
                "total_demand": total,
                "served_demand": served,
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
        return result
    finally:
        if project is not None:
            project.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        if owned_cache:
            shutil.rmtree(root, ignore_errors=True)
