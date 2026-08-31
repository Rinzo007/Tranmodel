"""AequilibraE-backed network pipeline."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from config import CACHE_DIR, LAYERS_DIR, PROJ_EPSG
from src.zones import build_transport_zones

AEQ_DIR = CACHE_DIR / "aequilibrae"
GMNS_DIR = AEQ_DIR / "gmns"
PROJECT_DIR = AEQ_DIR / "project"
TRANSIT_PROJECT_DIR = AEQ_DIR / "transit_project"
BUILD_VERSION = "tranmodel-aeq-v7-transit-support-compact"
VERSION_FILE = TRANSIT_PROJECT_DIR / "tranmodel_build_version.txt"
TRANSIT_SUPPORT_RADIUS_M = 900.0

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
    value = str(oneway).strip().lower() if oneway else ""
    if value in {"yes", "true", "1"}:
        return 1
    if value == "-1":
        return -1
    return 0


def _node_key(x: float, y: float) -> tuple[float, float]:
    return round(float(x), 7), round(float(y), 7)


def _filter_transit_support_roads(roads: gpd.GeoDataFrame, stops: gpd.GeoDataFrame,
                                  zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select roads close enough to transit stops and demand centroids."""
    roads_metric = roads.to_crs(PROJ_EPSG)
    anchors = list(stops.to_crs(PROJ_EPSG).geometry)
    anchors.extend(list(zones.to_crs(PROJ_EPSG).geometry.centroid))
    support_area = gpd.GeoSeries(anchors, crs=PROJ_EPSG).buffer(TRANSIT_SUPPORT_RADIUS_M).union_all()
    mask = roads_metric.geometry.intersects(support_area)
    return roads.loc[mask].copy()


def roads_to_gmns(roads: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert OSM road features to a compact GMNS topology."""
    notify = progress or (lambda _: None)
    roads = roads.to_crs("EPSG:4326").explode(index_parts=False, ignore_index=True)
    roads_metric = roads.to_crs(PROJ_EPSG)
    zones = zones.to_crs("EPSG:4326").copy()
    node_ids: dict[tuple[float, float], int] = {}
    node_rows: list[dict] = []
    link_rows: list[dict] = []

    def node_id(x: float, y: float) -> int:
        key = _node_key(x, y)
        if key in node_ids:
            return node_ids[key]
        nid = len(node_ids) + 1
        node_ids[key] = nid
        node_rows.append({"node_id": nid, "node_type": "", "x_coord": key[0], "y_coord": key[1]})
        return nid

    total = len(roads)
    empty = pd.Series(index=roads.index, dtype=object)
    notify(f"Строим компактный GMNS из {total:,} дорожных объектов...")
    for i, (geom, metric_geom, highway, maxspeed, oneway, lanes, name) in enumerate(zip(
        roads.geometry, roads_metric.geometry, roads.get("highway", empty), roads.get("maxspeed", empty),
        roads.get("oneway", empty), roads.get("lanes", empty), roads.get("name", empty),
    ), 1):
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            continue
        coords = list(geom.coords)
        if len(coords) < 2 or coords[0][:2] == coords[-1][:2]:
            continue
        a, b = coords[0], coords[-1]
        direction = _direction(oneway)
        if direction == -1:
            a, b = b, a
        a_id, b_id = node_id(a[0], a[1]), node_id(b[0], b[1])
        speed = _parse_speed(maxspeed, highway)
        try:
            lane_count = max(1.0, float(str(lanes).split(";")[0]))
        except (TypeError, ValueError):
            lane_count = DEFAULT_LANES
        length_m = float(metric_geom.length)
        if length_m <= 0:
            continue
        link_rows.append({
            "link_id": len(link_rows) + 1, "from_node_id": a_id, "to_node_id": b_id,
            "directed": int(direction != 0), "direction": direction, "length": length_m,
            "speed": speed, "capacity": lane_count * CAPACITY_PER_LANE, "lanes": lane_count,
            "link_type": str(highway or "unclassified"), "name": "" if pd.isna(name) else str(name),
            "modes": "c", "geometry": geom.wkt,
        })
        if i % 5000 == 0 or i == total:
            notify(f"GMNS: {i:,}/{total:,} объектов, {len(node_rows):,} узлов, {len(link_rows):,} связей")

    centroid_start = 9_000_000_000
    for pos, (_, row) in enumerate(zones.iterrows()):
        point = row.geometry.centroid
        node_rows.append({"node_id": centroid_start + pos + 1, "node_type": "centroid",
                          "x_coord": float(point.x), "y_coord": float(point.y)})
    return pd.DataFrame(node_rows), pd.DataFrame(link_rows)


def _write_gmns_files(nodes: pd.DataFrame, links: pd.DataFrame, force: bool, prefix: str = "") -> tuple[Path, Path]:
    GMNS_DIR.mkdir(parents=True, exist_ok=True)
    node_path = GMNS_DIR / f"{prefix}nodes.csv"
    link_path = GMNS_DIR / f"{prefix}links.csv"
    rewrite_links = force or not link_path.exists()
    if link_path.exists() and "directed" not in pd.read_csv(link_path, nrows=0).columns:
        rewrite_links = True
    if force or not node_path.exists():
        nodes.to_csv(node_path, index=False)
    if rewrite_links:
        links.to_csv(link_path, index=False)
    return link_path, node_path


def _project_is_current(project_dir: Path, version_file: Path) -> bool:
    if not project_dir.exists() or not version_file.exists():
        return False
    try:
        return version_file.read_text(encoding="utf-8").strip() == BUILD_VERSION
    except OSError:
        return False


def _build_from_gmns(project_dir: Path, link_path: Path, node_path: Path, nodes: pd.DataFrame, progress=None) -> Path:
    notify = progress or (lambda _: None)
    Project = _require_aequilibrae()
    project = Project()
    project.new(project_dir)
    project.network.create_from_gmns(link_file_path=str(link_path), node_file_path=str(node_path), srid=4326)
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
    notify("Проект AequilibraE готов и сохранён в кэше.")
    return project_dir


def build_project(force: bool = False, progress=None, *, mode: str = "transit") -> Path:
    """Build a cached AequilibraE support project."""
    notify = progress or (lambda _: None)
    if mode not in {"transit", "road"}:
        raise ValueError("mode must be 'transit' or 'road'")
    _require_aequilibrae()
    project_dir = TRANSIT_PROJECT_DIR if mode == "transit" else PROJECT_DIR
    version_file = VERSION_FILE if mode == "transit" else PROJECT_DIR / "tranmodel_build_version.txt"
    if not force and _project_is_current(project_dir, version_file):
        notify("Готовый проект AequilibraE найден в кэше — импорт дорожной сети не требуется.")
        return project_dir

    roads_path = LAYERS_DIR / "roads.parquet"
    if not roads_path.exists():
        raise FileNotFoundError(f"Road layer not found: {roads_path}")
    stops_path = CACHE_DIR / "phase1_real" / "stops_demand.parquet"
    if mode == "transit" and not stops_path.exists():
        raise FileNotFoundError(f"Stops layer not found: {stops_path}")
    if project_dir.exists():
        notify("Кэш проекта устарел — пересобираем поддержку AequilibraE...")
        shutil.rmtree(project_dir)
    if force and GMNS_DIR.exists():
        shutil.rmtree(GMNS_DIR)

    notify("Загружаем данные для проекта AequilibraE...")
    zones = build_transport_zones(force=False)
    roads = gpd.read_parquet(roads_path)
    if mode == "transit":
        stops = gpd.read_parquet(stops_path).to_crs(PROJ_EPSG)
        roads = _filter_transit_support_roads(roads, stops, zones)
        notify(f"Опорная дорожная сеть Transit: {len(roads):,} объектов (радиус {TRANSIT_SUPPORT_RADIUS_M:.0f} м).")

    nodes, links = roads_to_gmns(roads, zones, progress=notify)
    notify(f"GMNS готов: {len(nodes):,} узлов, {len(links):,} связей.")
    prefix = "transit_" if mode == "transit" else ""
    link_path, node_path = _write_gmns_files(nodes, links, force=force, prefix=prefix)
    notify(f"Создаём {mode}-проект AequilibraE: {len(nodes):,} узлов, {len(links):,} связей...")
    _build_from_gmns(project_dir, link_path, node_path, nodes, progress=notify)
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text(BUILD_VERSION, encoding="utf-8")
    return project_dir
