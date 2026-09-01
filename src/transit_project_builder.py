"""Fast direct-SQL builder for a thin AequilibraE transit project.

Bypasses ``create_from_gmns`` (which is quadratic due to
``correct_geometries``) by inserting nodes, links and zones directly.

The project is populated with:
- the full car-mode road network (nodes + links) from a pre-built
  networkx DiGraph, so AequilibraE's transit graph builder can compute
  realistic walking edges;
- stop-nodes (is_centroid=0) at snap coordinates;
- zone-centroids (is_centroid=1) at zone centroids, one per transport
  zone, matching the demand matrix shape.

All coordinates are converted to EPSG:4326 (lon/lat) before insertion:
road-graph nodes come in as (x, y) metres in ``PROJ_EPSG`` (UTM) and
zone centroids as kilometres in the same CRS.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, MultiPolygon, Point, box as shapely_box

from aequilibrae.project import Project
from aequilibrae.project.project_creation import add_triggers, remove_triggers

from config import PROJ_EPSG

_VERSION = "tranmodel-thin-transit-v2-lonlat"

_TO_WGS84 = Transformer.from_crs(f"EPSG:{PROJ_EPSG}", "EPSG:4326", always_xy=True)


def _to_lonlat(xy_metres: np.ndarray) -> np.ndarray:
    """Convert Nx2 array of (x, y) metres (PROJ_EPSG) to Nx2 (lon, lat)."""
    arr = np.asarray(xy_metres, dtype=float)
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i, 0], out[i, 1] = _TO_WGS84.transform(arr[i, 0], arr[i, 1])
    return out


def build_thin_transit_project(
    *,
    road_graph: nx.DiGraph,
    stop_mapping: list[tuple[float, float]],
    zone_centroids_xy: np.ndarray,
    stop_to_zone: dict[int, int],
    output_dir: Path,
) -> Path:
    """Build a thin AequilibraE project from the pre-built road graph.

    Parameters
    ----------
    road_graph : nx.DiGraph
        Road graph whose nodes are ``(x, y)`` coordinate tuples (metres,
        ``PROJ_EPSG`` / UTM).
    stop_mapping : list[tuple[float, float]]
        Road-graph node for each stop (same order as ``stop_xy`` in run.py),
        in metres (``PROJ_EPSG``).
    zone_centroids_xy : ndarray, shape (Z, 2), kilometres
        Zone centroid coordinates in ``PROJ_EPSG`` (km). Must match the
        demand matrix row/column order.
    stop_to_zone : dict[int, int]
        ``stop_index -> zone_index`` for every stop.
    output_dir : Path
        Where to create the AequilibraE project directory.

    Returns
    -------
    Path to the created project directory.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        vf = output_dir / "tranmodel_build_version.txt"
        if vf.exists() and vf.read_text(encoding="utf-8").strip() == _VERSION:
            print(f"build_thin_transit_project: cached project found at {output_dir}", flush=True)
            return output_dir
        shutil.rmtree(output_dir)

    t0 = time.perf_counter()

    # ---- identifiers ---------------------------------------------------
    graph_nodes = list(road_graph.nodes)                       # [(x,y), ...] metres
    road_node_id = {node: i + 1 for i, node in enumerate(graph_nodes)}
    n_road = len(graph_nodes)

    n_stops = len(stop_mapping)
    stop_start_id = n_road + 1
    stop_node_id = {i: stop_start_id + i for i in range(n_stops)}

    n_zones = len(zone_centroids_xy)
    zone_start_id = stop_start_id + n_stops
    zone_node_id = {i: zone_start_id + i for i in range(n_zones)}

    # ---- geometry conversion to EPSG:4326 ------------------------------
    road_xy_m = np.asarray(graph_nodes, dtype=float)                    # (N,2) metres
    road_lonlat = _to_lonlat(road_xy_m)                                 # (N,2) lon/lat

    stop_xy_m = np.asarray(stop_mapping, dtype=float)                   # (S,2) metres
    stop_lonlat = _to_lonlat(stop_xy_m) if n_stops else np.empty((0, 2))

    zone_xy_m = np.asarray(zone_centroids_xy, dtype=float) * 1000.0    # km -> metres
    zone_lonlat = _to_lonlat(zone_xy_m) if n_zones else np.empty((0, 2))

    # ---- create project ------------------------------------------------
    proj = Project()
    proj.new(str(output_dir))

    # ---- bulk insert (triggers off) ------------------------------------
    t_ins = time.perf_counter()
    with proj.db_connection as conn:
        remove_triggers(conn, proj.logger, "network")

        # --- road nodes (non-centroids) ---
        conn.executemany(
            "INSERT INTO nodes (node_id, is_centroid, geometry) VALUES (?, 0, GeomFromWKB(?, 4326))",
            [(nid, Point(float(x), float(y)).wkb) for node, nid in road_node_id.items()
             for x, y in [road_lonlat[nid - 1]]],
        )

        # --- stop-nodes (non-centroids, represent stops in the graph) ---
        conn.executemany(
            "INSERT INTO nodes (node_id, is_centroid, geometry) VALUES (?, 0, GeomFromWKB(?, 4326))",
            [(stop_node_id[i], Point(float(stop_lonlat[i, 0]), float(stop_lonlat[i, 1])).wkb)
             for i in range(n_stops)],
        )

        # --- zone-centroids (is_centroid=1, one per transport zone) ---
        conn.executemany(
            "INSERT INTO nodes (node_id, is_centroid, geometry) VALUES (?, 1, GeomFromWKB(?, 4326))",
            [(zone_node_id[i], Point(float(zone_lonlat[i, 0]), float(zone_lonlat[i, 1])).wkb)
             for i in range(n_zones)],
        )

        # --- road links (car mode 'c') ---
        link_id = 0
        link_rows: list[tuple] = []
        for u, v in road_graph.edges:
            link_id += 1
            ux, uy = road_lonlat[road_node_id[u] - 1]
            vx, vy = road_lonlat[road_node_id[v] - 1]
            ls = LineString([(float(ux), float(uy)), (float(vx), float(vy))])
            link_rows.append((
                link_id,
                road_node_id[u],
                road_node_id[v],
                0,       # direction (bidirectional; single arc in digraph)
                0,       # distance (auto-derived by triggers after add_triggers)
                "c",     # mode
                "default",
                ls.wkb,
            ))
        conn.executemany(
            "INSERT INTO links (link_id, a_node, b_node, direction, distance, modes, link_type, geometry) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, GeomFromWKB(?, 4326))",
            link_rows,
        )

        # --- zone polygons (tiny, for AequilibraE zoning) ---
        delta_deg = 0.0005  # ~50 m at Voronezh latitudes
        conn.executemany(
            "INSERT OR IGNORE INTO zones (zone_id, geometry) "
            "VALUES (?, ST_Multi(GeomFromWKB(?, 4326)))",
            [(zone_node_id[i],
              MultiPolygon([shapely_box(
                  float(zone_lonlat[i, 0]) - delta_deg,
                  float(zone_lonlat[i, 1]) - delta_deg,
                  float(zone_lonlat[i, 0]) + delta_deg,
                  float(zone_lonlat[i, 1]) + delta_deg,
              )]).wkb)
             for i in range(n_zones)],
        )

        add_triggers(conn, proj.logger, "network")
        conn.commit()

    # reload zoning in-memory
    from aequilibrae.project.zoning import Zoning
    proj.scenario.zoning = Zoning(proj.scenario.network)

    # write version file so build_project knows the project is current
    vf = output_dir / "tranmodel_build_version.txt"
    vf.write_text(_VERSION, encoding="utf-8")

    t_ins2 = time.perf_counter()
    proj.close()
    t_done = time.perf_counter()
    elapsed_ins = t_ins2 - t_ins
    elapsed_total = t_done - t0
    print(
        f"build_thin_transit_project: road_nodes={n_road:,}  road_links={link_id:,}  "
        f"stops={n_stops:,}  zones={n_zones:,}\n"
        f"  bulk insert: {elapsed_ins:.2f}s  total: {elapsed_total:.2f}s",
        flush=True,
    )
    return output_dir