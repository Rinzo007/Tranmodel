from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from typing import Any


def _frequency(route: Any) -> float:
    return float(getattr(route, "frequency_vph", getattr(route, "frequency", 6.0)))


def build_gtfs_from_route_set(route_set, stop_xy_lonlat, output_path: str | Path) -> Path:
    """Build a minimal valid GTFS feed for a TNDP candidate route set."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stops = {}
    for route in route_set.routes:
        for node in route.nodes:
            stops[int(node)] = True

    files = {
        "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\nTRANMODEL,Tranmodel,http://localhost,Europe/Moscow\n",
        "routes.txt": "route_id,agency_id,route_short_name,route_long_name,route_type\n"
        + "\n".join(f"R{i},TRANMODEL,{i+1},TNDP route {i+1},3" for i, _ in enumerate(route_set.routes))
        + "\n",
        "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\nWD,1,1,1,1,1,1,1,20260101,20261231\n",
    }
    stop_rows = ["stop_id,stop_name,stop_lat,stop_lon"]
    for node in sorted(stops):
        lon, lat = map(float, stop_xy_lonlat[node])
        stop_rows.append(f"S{node},Stop {node},{lat:.8f},{lon:.8f}")
    files["stops.txt"] = "\n".join(stop_rows) + "\n"

    trip_rows = ["route_id,service_id,trip_id"]
    stop_time_rows = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence"]
    frequencies = ["trip_id,start_time,end_time,headway_secs"]
    for i, route in enumerate(route_set.routes):
        trip_id = f"T{i}"
        files_route_id = f"R{i}"
        trip_rows.append(f"{files_route_id},WD,{trip_id}")
        for seq, node in enumerate(route.nodes):
            hour = 6 + min(seq // 60, 17)
            minute = seq % 60
            t = f"{hour:02d}:{minute:02d}:00"
            stop_time_rows.append(f"{trip_id},{t},{t},S{int(node)},{seq+1}")
        headway = max(300, int(round(3600.0 / max(_frequency(route), 0.1))))
        frequencies.append(f"{trip_id},06:00:00,23:00:00,{headway}")
    files["trips.txt"] = "\n".join(trip_rows) + "\n"
    files["stop_times.txt"] = "\n".join(stop_time_rows) + "\n"
    files["frequencies.txt"] = "\n".join(frequencies) + "\n"

    with ZipFile(output_path, "w", ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return output_path
