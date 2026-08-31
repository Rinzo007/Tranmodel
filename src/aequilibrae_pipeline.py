"""AequilibraE-backed network pipeline.

The road graph is independent from demand zones and public-transport stops.
Transport zones provide OD centroids; transit stops remain transit entities.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from config import CACHE_DIR, LAYERS_DIR
from src.zones import build_transport_zones

AEQ_DIR = CACHE_DIR / "aequilibrae"
GMNS_DIR = AEQ_DIR / "gmns"
PROJECT_DIR = AEQ_DIR / "project"

DEFAULT_SPEED_KMH = {
    "motorway": 90.0, "motorway_link": 50.0, "trunk": 70.0, "trunk_link": 40.0,
    "primary": 60.0, "primary_link": 35.0, "secondary": 50.0, "secondary_link": 30.0,
    "tertiary": 40.0, "tertiary_link": 25.0, "unclassified": 30.0, "residential": 30.0,
    "living_street": 20.0, "service": 20.0, "road": 30.0, "track": 15.0,
    "pedestrian": 5.0, "footway": 5.0, "cycleway": 15.0, "services": 20.0,
}
DEFAULT_LANES = 1.0
CAPACITY_PER_LANE = 900.0


class AequilibraEPipelineError(RuntimeError):
    pass


def _require_aequilibrae():
    try:
        from aequilibrae import Project
        return Project
    except ImportError as exc:
        raise AequilibraEPipelineError("AequilibraE is not installed. Install requirements.txt first.") from exc


def _parse_speed(value, highway: str | None) -> float:
    if value is not None and not pd.isna(value):
        match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", "."))
        if match:
            speed = float(match.group(1))
            if 1.0 <= speed <= 160.0:
                return speed
    return DEFAULT_SPEED_KMH.get(str(highway or "").lower(), 30.0)


def _direction(oneway) -> int:
    if oneway:
        value = str(oneway).strip().lower()
    else:
        value = ""
    if value in {"yes", "true", "1"}:
        return 1
    if value == "-1":
        return -1
    return 0


def _node_key(x: float, y: float) -> tuple[float, float]:
    return round(float(x), 7), round(float(y), 7)


def roads_to_gmns(roads: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the real road network and zone centroids to GMNS."""
    roads = roads.to_crs("EPSG:4326").explode(index_parts=False, ignore_index=True)
    zones = zones.to_crs("EPSG:4326").copy()
    node_ids: dict[tuple[float, float], int] = {}
    node_rows: list[dict] = []
    link_rows: list[dict] = []

    def node_id(x: float, y: float) -> int:
        key = _node_key(x, y)
        existing = node_ids.get(key)
        if existing is not None:
            return existing
        nid = len(node_ids) + 1
        node_ids[key] = nid
        node_rows.append({"node_id": nid, "node_type": "", "x_coord": key[0], "y_coord": key[1]})
        return nid

    next_link = 1
    empty = pd.Series(index=roads.index, dtype=object)
    for geom, highway, maxspeed, oneway, lanes, name in zip(
        roads.geometry, roads.get("highway", empty), roads.get("maxspeed", empty),
        roads.get("oneway", empty), roads.get("lanes", empty), roads.get("name", empty),
    ):
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            continue
        pts = list(geom.coords)
        for a, b in zip(pts[:-1], pts[1:]):
            if a[:2] == b[:2]:
                continue
            d = _direction(oneway)
            if d == -1:
                a, b, d = b, a, 1
            a_id, b_id = node_id(a[0], a[1]), node_id(b[0], b[1])
            speed = _parse_speed(maxspeed, highway)
            try:
                lane_count = max(1.0, float(str(lanes).split(";")[0]))
            except (TypeError, ValueError):
                lane_count = DEFAULT_LANES
            length_m = float(LineString([a, b]).length)
            link_rows.append({
                "link_id": next_link,
                "from_node_id": a_id,
                "to_node_id": b_id,
                "directed": int(d != 0),
                "direction": d,
                "length": length_m,
                "speed": speed,
                "capacity": lane_count * CAPACITY_PER_LANE,
                "lanes": lane_count,
                "link_type": str(highway or "unclassified"),
                "name": "" if pd.isna(name) else str(name),
                "modes": "c",
                "geometry": LineString([a, b]).wkt,
            })
            next_link += 1

    centroid_start = 9_000_000_000
    for pos, (_, row) in enumerate(zones.iterrows()):
        point = row.geometry.centroid
        node_rows.append({
            "node_id": centroid_start + pos + 1,
            "node_type": "centroid",
            "x_coord": float(point.x),
            "y_coord": float(point.y),
        })
    return pd.DataFrame(node_rows), pd.DataFrame(link_rows)


def _write_gmns_files(nodes: pd.DataFrame, links: pd.DataFrame, force: bool) -> tuple[Path, Path]:
    GMNS_DIR.mkdir(parents=True, exist_ok=True)
    node_path, link_path = GMNS_DIR / "nodes.csv", GMNS_DIR / "links.csv"
    rewrite_links = force or not link_path.exists()
    if link_path.exists():
        existing_columns = pd.read_csv(link_path, nrows=0).columns
        if "directed" not in existing_columns:
            rewrite_links = True
    rewrite_nodes = force or not node_path.exists()
    if rewrite_nodes:
        nodes.to_csv(node_path, index=False)
    if rewrite_links:
        links.to_csv(link_path, index=False)
    return link_path, node_path


def build_project(force: bool = False) -> Path:
    """Create/open the AequilibraE road project using transport-zone centroids."""
    Project = _require_aequilibrae()
    roads_path = LAYERS_DIR / "roads.parquet"
    if not roads_path.exists():
        raise FileNotFoundError(f"Road layer not found: {roads_path}")

    zones = build_transport_zones(force=False)
    roads = gpd.read_parquet(roads_path)
    if force and PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    nodes, links = roads_to_gmns(roads, zones)
    link_path, node_path = _write_gmns_files(nodes, links, force=force)

    if not PROJECT_DIR.exists():
        project = Project()
        project.new(PROJECT_DIR)
        project.network.create_from_gmns(
            link_file_path=str(link_path), node_file_path=str(node_path), srid=4326,
        )
        centroid_ids = nodes.loc[nodes["node_type"] == "centroid", "node_id"].astype(int)
        for cid in centroid_ids:
            node = project.network.nodes.get(int(cid))
            try:
                node.connect_mode("c", connectors=3, limit_to_zone=False)
            except TypeError:
                node.connect_mode("c", connectors=3)
        project.network.nodes.save()
        project.network.links.save()
        project.close()
    return PROJECT_DIR
