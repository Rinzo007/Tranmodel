"""Public-transport integration for the real Voronezh route reference.

The repository GeoJSON is converted to a minimal GTFS feed and imported into
AequilibraE's public-transport database. The same OD matrix produced by the
AequilibraE distribution step is then assigned on the transit graph using
Optimal Strategies (Spiess & Florian).

The reference file does not contain service calendars, headways or travel
speeds, so those values are explicit model assumptions and are reported as
such. Route numbers, route membership, terminals and stop coordinates come
from ``voronezh_routes_terminals.geojson``.
"""

from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np

from config import CACHE_DIR, REFERENCE_ROUTES_PATH, REPORT_DIR

AEQ_DIR = CACHE_DIR / "aequilibrae"
PT_DIR = AEQ_DIR / "transit"
GTFS_DIR = PT_DIR / "gtfs"
GTFS_ZIP = PT_DIR / "voronezh_reference_gtfs.zip"

# Explicit assumptions because the reference GeoJSON has no timetable data.
PT_MODE = 3  # GTFS bus
PT_HEADWAY_MIN = 10.0
PT_SERVICE_START = 6 * 3600
PT_SERVICE_END = 23 * 3600
PT_SPEED_KMH = 22.0
PT_DWELL_SEC = 20
PT_WALKING_SPEED_KMH = 4.5
PT_MAX_CONNECTOR_M = 800.0
PT_MAX_CONNECTORS_PER_ZONE = 3
GTFS_SERVICE_DATE = "2026-01-15"
GTFS_VALID_FROM = "20260101"
GTFS_VALID_TO = "20261231"


class TransitPipelineError(RuntimeError):
    """Raised when the public transport pipeline cannot be completed."""


def _import_phase3_helpers():
    from src.phase3_real import _build_routes, _match_stops
    return _build_routes, _match_stops


def load_reference_routes() -> gpd.GeoDataFrame:
    if not REFERENCE_ROUTES_PATH.exists():
        raise FileNotFoundError(f"Reference route file not found: {REFERENCE_ROUTES_PATH}")
    ref = gpd.read_file(REFERENCE_ROUTES_PATH)
    if ref.empty:
        raise TransitPipelineError("Reference route GeoJSON is empty")
    if "routes" not in ref.columns:
        raise TransitPipelineError("Reference GeoJSON must contain a 'routes' property")
    return ref.to_crs("EPSG:4326")


def _load_participating_stops() -> gpd.GeoDataFrame:
    path = CACHE_DIR / "phase1_real" / "stops_demand.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Demand stops not found: {path}. Run phase 1 real first.")
    stops = gpd.read_parquet(path).to_crs("EPSG:4326").reset_index(drop=True)
    if stops.empty:
        raise TransitPipelineError("No participating stops")
    return stops


def build_route_sequences() -> tuple[dict[str, list[int]], gpd.GeoDataFrame]:
    """Return route number -> ordered stop indices in phase1_real stop table."""
    ref = load_reference_routes()
    stops = _load_participating_stops()
    build_routes, match_stops = _import_phase3_helpers()
    matched = match_stops(ref, stops, max_snap_m=100.0)
    all_routes, route_info = build_routes(matched, stops)
    route_ids = list(route_info.keys())
    if len(route_ids) != len(all_routes):
        raise TransitPipelineError(
            "Reference route metadata and ordered route list have different lengths"
        )
    sequences = {}
    for rid, (ordered, _start, _end) in zip(route_ids, all_routes):
        key = str(rid)
        clean = []
        seen = set()
        for idx in ordered:
            idx = int(idx)
            if idx not in seen:
                clean.append(idx)
                seen.add(idx)
        if len(clean) >= 2:
            sequences[key] = clean
    return sequences, stops


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    lat = math.radians((lat1 + lat2) / 2.0)
    dx = math.radians(lon2 - lon1) * 6371000.0 * math.cos(lat)
    dy = math.radians(lat2 - lat1) * 6371000.0
    return float(math.hypot(dx, dy))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_gtfs(force: bool = False) -> Path:
    """Create a self-contained GTFS feed from the repository route reference."""
    if GTFS_ZIP.exists() and not force:
        return GTFS_ZIP

    PT_DIR.mkdir(parents=True, exist_ok=True)
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    for p in GTFS_DIR.glob("*.txt"):
        p.unlink()

    seqs, stops = build_route_sequences()
    if not seqs:
        raise TransitPipelineError("No valid routes could be built from reference GeoJSON")

    stop_rows = []
    used_stop_ids = sorted({idx for seq in seqs.values() for idx in seq})
    for idx in used_stop_ids:
        row = stops.iloc[idx]
        stop_rows.append(
            {
                "stop_id": f"s{idx + 1}",
                "stop_name": str(row.get("name") or f"Остановка {idx + 1}"),
                "stop_lat": float(row.geometry.y),
                "stop_lon": float(row.geometry.x),
                "location_type": 0,
            }
        )
    stop_meta = {r["stop_id"]: r for r in stop_rows}

    agencies = [{
        "agency_id": "VORONEZH",
        "agency_name": "Воронежская маршрутная сеть",
        "agency_url": "https://www.openstreetmap.org/",
        "agency_timezone": "Europe/Moscow",
        "agency_lang": "ru",
    }]
    feed_info = [{
        "feed_publisher_name": "Tranmodel",
        "feed_publisher_url": "https://github.com/Rinzo007/Tranmodel",
        "feed_lang": "ru",
        "feed_start_date": GTFS_VALID_FROM,
        "feed_end_date": GTFS_VALID_TO,
        "feed_version": "reference-1",
    }]
    routes_rows = []
    calendar_rows = [{
        "service_id": "weekday",
        "monday": 1, "tuesday": 1, "wednesday": 1,
        "thursday": 1, "friday": 1, "saturday": 1, "sunday": 1,
        "start_date": GTFS_VALID_FROM,
        "end_date": GTFS_VALID_TO,
    }]
    trips_rows, stop_times_rows, shapes_rows = [], [], []

    shape_counter = 1
    trip_counter = 1
    route_index = 1
    departures = np.arange(
        PT_SERVICE_START, PT_SERVICE_END + 1, int(PT_HEADWAY_MIN * 60)
    )

    for rid in sorted(seqs, key=lambda x: (len(x), x)):
        seq = seqs[rid]
        route_id = f"r{route_index}"
        route_index += 1
        routes_rows.append({
            "route_id": route_id,
            "agency_id": "VORONEZH",
            "route_short_name": str(rid),
            "route_long_name": f"Маршрут {rid}",
            "route_type": PT_MODE,
        })

        for direction_id, directed in enumerate((seq, list(reversed(seq)))):
            shape_id = f"sh{shape_counter}"
            shape_counter += 1
            for order, idx in enumerate(directed):
                row = stops.iloc[idx]
                shapes_rows.append({
                    "shape_id": shape_id,
                    "shape_pt_lat": float(row.geometry.y),
                    "shape_pt_lon": float(row.geometry.x),
                    "shape_pt_sequence": order + 1,
                })

            for dep in departures:
                trip_id = f"t{trip_counter}"
                trip_counter += 1
                trips_rows.append({
                    "route_id": route_id,
                    "service_id": "weekday",
                    "trip_id": trip_id,
                    "trip_headsign": stop_meta[f"s{directed[-1] + 1}"]["stop_name"],
                    "direction_id": direction_id,
                    "shape_id": shape_id,
                })
                elapsed = 0.0
                for stop_seq, idx in enumerate(directed):
                    if stop_seq > 0:
                        prev = stops.iloc[directed[stop_seq - 1]]
                        cur = stops.iloc[idx]
                        dist_m = _distance_m(
                            (float(prev.geometry.x), float(prev.geometry.y)),
                            (float(cur.geometry.x), float(cur.geometry.y)),
                        )
                        elapsed += dist_m / (PT_SPEED_KMH * 1000.0 / 3600.0)
                        elapsed += PT_DWELL_SEC
                    arr = int(dep + elapsed)
                    hh, rem = divmod(arr, 3600)
                    mm, ss = divmod(rem, 60)
                    tm = f"{hh:02d}:{mm:02d}:{ss:02d}"
                    stop_times_rows.append({
                        "trip_id": trip_id,
                        "arrival_time": tm,
                        "departure_time": tm,
                        "stop_id": f"s{idx + 1}",
                        "stop_sequence": stop_seq + 1,
                    })

    _write_csv(GTFS_DIR / "agency.txt", list(agencies[0].keys()), agencies)
    _write_csv(GTFS_DIR / "feed_info.txt", list(feed_info[0].keys()), feed_info)
    _write_csv(GTFS_DIR / "stops.txt", list(stop_rows[0].keys()), stop_rows)
    _write_csv(GTFS_DIR / "routes.txt", list(routes_rows[0].keys()), routes_rows)
    _write_csv(GTFS_DIR / "calendar.txt", list(calendar_rows[0].keys()), calendar_rows)
    _write_csv(GTFS_DIR / "trips.txt", list(trips_rows[0].keys()), trips_rows)
    _write_csv(GTFS_DIR / "stop_times.txt", list(stop_times_rows[0].keys()), stop_times_rows)
    _write_csv(GTFS_DIR / "shapes.txt", list(shapes_rows[0].keys()), shapes_rows)

    with zipfile.ZipFile(GTFS_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(GTFS_DIR.glob("*.txt")):
            zf.write(p, p.name)
    return GTFS_ZIP


def _make_assignment_matrix_from_existing(demand_matrix):
    """Copy the gravity matrix into a transit-class matrix with the same zone IDs."""
    from aequilibrae.matrix import AequilibraeMatrix

    raw = demand_matrix.get_matrix("gravity")
    source_ids = np.asarray(demand_matrix.index, dtype=np.int64)
    matrix = AequilibraeMatrix()
    matrix.create_empty(zones=len(source_ids), matrix_names=["pt"], memory_only=True)
    matrix.index = source_ids
    matrix.matrices[:, :, 0] = np.asarray(raw, dtype=float)
    matrix.computational_view(["pt"])
    return matrix


def run_transit_assignment(project, demand_matrix, force: bool = False) -> dict:
    """Import the reference routes and assign OD demand to public transport."""
    from aequilibrae.paths import TransitAssignment, TransitClass
    from aequilibrae.transit import Transit

    gtfs_path = build_gtfs(force=force)
    data = Transit(project)
    imported = data.new_gtfs_builder(
        agency="VORONEZH",
        file_path=str(gtfs_path),
        day=GTFS_SERVICE_DATE,
        description="Voronezh reference route network generated from GeoJSON",
    )
    imported.load_date(GTFS_SERVICE_DATE)
    data.save_to_disk()

    graph_builder = data.create_graph(
        with_inner_stop_transfers=True,
        with_outer_stop_transfers=False,
        with_walking_edges=True,
        distance_upper_bound=PT_MAX_CONNECTOR_M,
        blocking_centroid_flows=True,
        connector_method="nearest_neighbour",
        max_connectors_per_zone=PT_MAX_CONNECTORS_PER_ZONE,
    )
    try:
        graph_builder.create_line_geometry(method="connector project match", graph="c")
    except Exception:
        graph_builder.create_line_geometry(method="direct")
    graph_builder.save()
    transit_graph = graph_builder.to_transit_graph()

    source_ids = set(map(int, demand_matrix.index))
    taz_ids = set(map(int, graph_builder.od_node_mapping["taz_id"]))
    if not source_ids.issubset(taz_ids):
        missing = sorted(source_ids - taz_ids)[:10]
        raise TransitPipelineError(f"Transit graph is missing demand zones, examples: {missing}")

    demand = _make_assignment_matrix_from_existing(demand_matrix)
    transit_class = TransitClass(name="pt", graph=transit_graph, matrix=demand)
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

    load = transit_class.results.get_load_results().reset_index()
    PT_DIR.mkdir(parents=True, exist_ok=True)
    load.to_parquet(PT_DIR / "transit_link_load.parquet", index=False)

    skim_results = assignment.get_skim_results()["pt"].matrix
    for field in ("trav_time", "waiting_time", "transfers", "boardings"):
        if field in skim_results:
            np.save(PT_DIR / f"skim_{field}.npy", np.asarray(skim_results[field]))

    report = {
        "backend": "AequilibraE 1.7.0 transit",
        "gtfs": str(gtfs_path),
        "n_routes": int(data.get_table("routes").shape[0]),
        "n_stops": int(data.get_table("stops").shape[0]),
        "n_route_links": int(data.get_table("route_links").shape[0]),
        "n_transit_graph_nodes": int(len(transit_graph.vertices)),
        "n_transit_graph_edges": int(len(transit_graph.edges)),
        "n_centroids": int(len(transit_graph.centroids)),
        "total_demand": float(np.asarray(demand.get_matrix("pt")).sum()),
        "assigned_link_rows": int(len(load)),
        "headway_min": PT_HEADWAY_MIN,
        "service_start": PT_SERVICE_START,
        "service_end": PT_SERVICE_END,
        "speed_kmh": PT_SPEED_KMH,
        "dwell_sec": PT_DWELL_SEC,
        "walking_speed_kmh": PT_WALKING_SPEED_KMH,
        "connector_max_m": PT_MAX_CONNECTOR_M,
        "assignment": "Optimal Strategies",
        "reference_source": REFERENCE_ROUTES_PATH.name,
        "assumptions": {
            "headway": "uniform 10 min because source GeoJSON has no timetable/frequency",
            "speed": "22 km/h because source GeoJSON has no running times",
            "calendar": "daily service 06:00-23:00 because source GeoJSON has no calendar",
        },
    }
    (PT_DIR / "transit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORT_DIR / "transit_report.md").write_text(
        "\n".join([
            "# Общественный транспорт — AequilibraE",
            "",
            f"- Маршрутов: **{report['n_routes']:,}**",
            f"- Остановок: **{report['n_stops']:,}**",
            f"- Рёбер маршрутов: **{report['n_route_links']:,}**",
            f"- Зон-центроидов: **{report['n_centroids']:,}**",
            f"- Спрос: **{report['total_demand']:,.0f}** поездок/сутки",
            "- Назначение: **Optimal Strategies (Spiess & Florian)**",
            f"- Средний интервал по умолчанию: **{report['headway_min']:.0f} мин**",
            f"- Скорость в движении: **{report['speed_kmh']:.0f} км/ч**",
            "",
            "## Источник",
            f"`{report['reference_source']}`",
            "",
            "## Допущения",
            "В GeoJSON отсутствуют расписания и частоты, поэтому интервал, скорость и календарь заданы параметрами модели.",
        ]),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    raise SystemExit(
        "Run this module through aequilibrae_pipeline.run_all(); it needs a prepared AequilibraE project and demand matrix."
    )
