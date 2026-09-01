from __future__ import annotations

from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import networkx as nx
from pyproj import Transformer

from config import PROJ_EPSG

_PROJECTED_TO_WGS84 = Transformer.from_crs(PROJ_EPSG, "EPSG:4326", always_xy=True)


def _frequency(route: Any) -> float:
    return float(getattr(route, "frequency_vph", getattr(route, "frequency", 6.0)))


def _fmt_time(seconds: int) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _path_between_stops(road_graph: nx.Graph, stop_mapping, a: int, b: int, path_index=None):
    if path_index is not None:
        cached = path_index.get(int(a), int(b))
        if cached is not None:
            return cached
    ra, rb = stop_mapping[int(a)], stop_mapping[int(b)]
    path = tuple(nx.shortest_path(road_graph, ra, rb, weight="time"))
    time_min = float(nx.path_weight(road_graph, path, weight="time"))
    length_km = float(nx.path_weight(road_graph, path, weight="length_km"))
    return path, time_min, length_km


def build_gtfs_from_route_set(route_set, stop_xy_lonlat, output_path: str | Path, *, road_graph: nx.Graph | None = None, stop_mapping=None, path_index=None) -> Path:
    """Build GTFS from a candidate route set using cached real-road paths."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if road_graph is None or stop_mapping is None:
        raise ValueError("road_graph and stop_mapping are required for GTFS generation")

    uses_stops = sorted({int(node) for route in route_set.routes for node in route.nodes})
    files: dict[str, str] = {
        "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\nTRANMODEL,Tranmodel,http://localhost,Europe/Moscow\n",
        "routes.txt": "route_id,route_short_name,route_long_name,route_type\n" + "\n".join(f"R{i},{i + 1},TNDP route {i + 1},3" for i, _ in enumerate(route_set.routes)) + "\n",
        "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\nWD,1,1,1,1,1,1,1,20260101,20261231\n",
    }
    stop_rows = ["stop_id,stop_name,stop_lat,stop_lon"]
    for node in uses_stops:
        lon, lat = map(float, stop_xy_lonlat[node])
        stop_rows.append(f"S{node},Stop {node},{lat:.8f},{lon:.8f}")
    files["stops.txt"] = "\n".join(stop_rows) + "\n"

    trip_rows = ["route_id,service_id,trip_id,trip_headsign,shape_id,direction_id"]
    stop_time_rows = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence"]
    shape_rows = ["shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence"]

    first_departure_sec = 6 * 3600
    service_end_sec = 23 * 3600
    for i, route in enumerate(route_set.routes):
        trip_id, route_id, shape_id = f"T{i}", f"R{i}", f"SH{i}"
        headway = max(300, int(round(3600.0 / max(_frequency(route), 0.1))))
        first = int(route.nodes[0])
        shape_sequence = 1
        lon, lat = map(float, stop_xy_lonlat[first])
        shape_rows.append(f"{shape_id},{lat:.8f},{lon:.8f},{shape_sequence}")
        shape_sequence += 1

        segment_times_min: list[float] = []
        for a, b in zip(route.nodes[:-1], route.nodes[1:]):
            path, segment_time_min, _ = _path_between_stops(road_graph, stop_mapping, a, b, path_index)
            segment_times_min.append(segment_time_min)
            for road_node in path[1:]:
                x, y = map(float, road_node)
                lon, lat = _PROJECTED_TO_WGS84.transform(x, y)
                shape_rows.append(f"{shape_id},{lat:.8f},{lon:.8f},{shape_sequence}")
                shape_sequence += 1

        travel_sec = sum(max(1, int(round(t * 60.0))) for t in segment_times_min)

        dep = first_departure_sec
        trip_index = 0
        while dep < service_end_sec:
            trip_n = f"{trip_id}-{trip_index}"
            trip_rows.append(f"{route_id},WD,{trip_n},TNDP route {i + 1},{shape_id},0")
            current_seconds = dep
            stop_time_rows.append(f"{trip_n},{_fmt_time(current_seconds)},{_fmt_time(current_seconds)},S{first},1")
            for seq, a in enumerate(route.nodes[1:], start=2):
                current_seconds += max(1, int(round(segment_times_min[seq - 2] * 60.0)))
                t = _fmt_time(current_seconds)
                stop_time_rows.append(f"{trip_n},{t},{t},S{int(a)},{seq}")
            dep += headway
            trip_index += 1
        assert trip_index > 0, f"no trips generated for route {route_id}"

    files["trips.txt"] = "\n".join(trip_rows) + "\n"
    files["stop_times.txt"] = "\n".join(stop_time_rows) + "\n"
    files["shapes.txt"] = "\n".join(shape_rows) + "\n"
    with ZipFile(output_path, "w", ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return output_path
