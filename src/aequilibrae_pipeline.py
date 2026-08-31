"""AequilibraE-backed transport-model pipeline.

This module replaces the custom networkx/Dijkstra/Furness/route-planner core with
AequilibraE primitives while keeping the existing OSM + WorldPop preprocessing.

Workflow:
  1. Convert cached OSM roads + real/reference stops into an AequilibraE project.
  2. Treat transit stops as demand centroids and connect them to the road graph.
  3. Compute network travel-time/distance skims with AequilibraE.
  4. Apply AequilibraE synthetic gravity model (EXPO) and internal IPF/Furness.
  5. Run equilibrium traffic assignment (BFW/BPR) and export link loads.

The resulting project is stored in data/cache/aequilibrae/ and can be opened
from QAequilibraE/QGIS as a normal AequilibraE project.
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

from config import CACHE_DIR, LAYERS_DIR, PROJ_EPSG, REPORT_DIR

AEQ_DIR = CACHE_DIR / "aequilibrae"
GMNS_DIR = AEQ_DIR / "gmns"
PROJECT_DIR = AEQ_DIR / "project"

# Old model used exp(-distance / 5.5 km).  We use network travel time instead.
DECAY_RADIUS_KM = 5.5
REFERENCE_SPEED_KMH = 25.0
EXPONENTIAL_BETA = 1.0 / (DECAY_RADIUS_KM / REFERENCE_SPEED_KMH * 60.0)

DEFAULT_SPEED_KMH = {
    "motorway": 90.0,
    "motorway_link": 50.0,
    "trunk": 70.0,
    "trunk_link": 40.0,
    "primary": 60.0,
    "primary_link": 35.0,
    "secondary": 50.0,
    "secondary_link": 30.0,
    "tertiary": 40.0,
    "tertiary_link": 25.0,
    "unclassified": 30.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
    "road": 30.0,
    "track": 15.0,
    "pedestrian": 5.0,
    "footway": 5.0,
    "cycleway": 15.0,
    "services": 20.0,
}

DEFAULT_LANES = 1.0
CAPACITY_PER_LANE = 900.0


class AequilibraEPipelineError(RuntimeError):
    """Raised when an AequilibraE pipeline step cannot be completed."""


def _require_aequilibrae():
    try:
        from aequilibrae import Project
        return Project
    except ImportError as exc:
        raise AequilibraEPipelineError(
            "AequilibraE is not installed. Install requirements.txt first."
        ) from exc


def _parse_speed(value, highway: str | None) -> float:
    if value is not None and not pd.isna(value):
        match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", "."))
        if match:
            speed = float(match.group(1))
            if 1.0 <= speed <= 160.0:
                return speed
    return DEFAULT_SPEED_KMH.get(str(highway or "").lower(), 30.0)


def _direction(oneway) -> int:
    if oneway is None or pd.isna(oneway):
        return 0
    value = str(oneway).strip().lower()
    if value in {"yes", "true", "1"}:
        return 1
    if value == "-1":
        return -1
    return 0


def _explode_lines(roads: gpd.GeoDataFrame) -> list[LineString]:
    lines: list[LineString] = []
    for geom in roads.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            if len(geom.coords) >= 2:
                lines.append(geom)
        elif geom.geom_type == "MultiLineString":
            lines.extend(x for x in geom.geoms if len(x.coords) >= 2)
    return lines


def _node_key(x: float, y: float) -> tuple[float, float]:
    # OSM coordinates normally match exactly; rounding also prevents tiny
    # floating-point differences from producing disconnected duplicate nodes.
    return round(float(x), 7), round(float(y), 7)


def roads_to_gmns(roads: gpd.GeoDataFrame, stops: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert cached road lines + demand stops to GMNS tables."""
    roads = roads.to_crs("EPSG:4326").explode(index_parts=False, ignore_index=True)
    stops = stops.to_crs("EPSG:4326").copy()

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
        node_rows.append({
            "node_id": nid,
            "node_type": "",
            "x_coord": key[0],
            "y_coord": key[1],
        })
        return nid

    next_link = 1
    for geom, highway, maxspeed, oneway, lanes, name in zip(
        roads.geometry,
        roads.get("highway", pd.Series(index=roads.index, dtype=object)),
        roads.get("maxspeed", pd.Series(index=roads.index, dtype=object)),
        roads.get("oneway", pd.Series(index=roads.index, dtype=object)),
        roads.get("lanes", pd.Series(index=roads.index, dtype=object)),
        roads.get("name", pd.Series(index=roads.index, dtype=object)),
    ):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type != "LineString" or len(geom.coords) < 2:
            continue
        pts = list(geom.coords)
        for a, b in zip(pts[:-1], pts[1:]):
            if a[:2] == b[:2]:
                continue
            a_id = node_id(a[0], a[1])
            b_id = node_id(b[0], b[1])
            d = _direction(oneway)
            speed = _parse_speed(maxspeed, highway)
            lane_count = DEFAULT_LANES
            try:
                lane_count = max(1.0, float(str(lanes).split(";")[0]))
            except (TypeError, ValueError):
                pass
            link_rows.append({
                "link_id": next_link,
                "from_node_id": a_id,
                "to_node_id": b_id,
                "direction": 1 if d == -1 else d,
                "length": float(LineString([a, b]).length),
                "speed": speed,
                "capacity": lane_count * CAPACITY_PER_LANE,
                "lanes": lane_count,
                "link_type": str(highway or "unclassified"),
                "name": "" if pd.isna(name) else str(name),
                "modes": "car",
                "geometry": LineString([a, b]).wkt if d != -1 else LineString([b, a]).wkt,
            })
            next_link += 1

    # Demand stops become centroids.  Keep their IDs separate from physical nodes.
    centroid_rows: list[dict] = []
    centroid_start = 9_000_000_000
    stops = stops.reset_index(drop=True)
    for pos, row in stops.iterrows():
        geom = row.geometry
        centroid_rows.append({
            "node_id": centroid_start + pos,
            "node_type": "centroid",
            "x_coord": float(geom.x),
            "y_coord": float(geom.y),
        })
    node_rows.extend(centroid_rows)

    return pd.DataFrame(node_rows), pd.DataFrame(link_rows)


def _write_gmns_files(nodes: pd.DataFrame, links: pd.DataFrame, force: bool) -> tuple[Path, Path]:
    GMNS_DIR.mkdir(parents=True, exist_ok=True)
    node_path = GMNS_DIR / "nodes.csv"
    link_path = GMNS_DIR / "links.csv"
    if force or not node_path.exists():
        nodes.to_csv(node_path, index=False)
    if force or not link_path.exists():
        links.to_csv(link_path, index=False)
    return link_path, node_path


def build_project(force: bool = False) -> Path:
    """Create/open an AequilibraE project from the cached model layers."""
    Project = _require_aequilibrae()
    roads_path = LAYERS_DIR / "roads.parquet"
    stops_path = CACHE_DIR / "phase1_real" / "stops_demand.parquet"
    if not roads_path.exists():
        raise FileNotFoundError(f"Road layer not found: {roads_path}")
    if not stops_path.exists():
        raise FileNotFoundError(
            f"Demand stops not found: {stops_path}. Run phase 1 real first."
        )

    if force and PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    PROJECT_DIR.parent.mkdir(parents=True, exist_ok=True)

    roads = gpd.read_parquet(roads_path)
    stops = gpd.read_parquet(stops_path)
    nodes, links = roads_to_gmns(roads, stops)
    link_path, node_path = _write_gmns_files(nodes, links, force=force)

    if not PROJECT_DIR.exists():
        project = Project()
        project.new(PROJECT_DIR)
        project.network.create_from_gmns(
            link_file_path=str(link_path),
            node_file_path=str(node_path),
            srid=4326,
        )

        # Connect every demand centroid to the nearest car node. AequilibraE
        # then handles centroid connectors and centroid-aware shortest paths.
        centroid_ids = nodes.loc[nodes["node_type"] == "centroid", "node_id"].astype(int)
        for cid in centroid_ids:
            node = project.network.nodes.get(int(cid))
            try:
                node.connect_mode("c", connectors=1, limit_to_zone=False)
            except TypeError:
                # Compatibility fallback for older AequilibraE releases.
                node.connect_mode("c", connectors=1)
        project.network.nodes.save()
        project.network.links.save()
        project.close()

    return PROJECT_DIR


def _open_project():
    Project = _require_aequilibrae()
    return Project.from_path(PROJECT_DIR)


def run_skimming(project) -> object:
    """Build car graph and return an AequilibraE skim matrix."""
    from aequilibrae.paths import NetworkSkimming

    project.network.build_graphs(modes=["c"])
    graph = project.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_skimming(["free_flow_time", "distance"])
    graph.set_blocked_centroid_flows(True)

    skm = NetworkSkimming(graph)
    skm.execute()
    if skm.report:
        raise AequilibraEPipelineError("AequilibraE skimming failed: " + "; ".join(map(str, skm.report)))
    skm.save_to_project("network_skims", "omx")
    return skm.results.skims


def run_gravity(project, impedance) -> tuple[object, pd.DataFrame]:
    """Apply an exponential synthetic gravity model to network-time impedance."""
    from aequilibrae.distribution import GravityApplication, SyntheticGravityModel

    impedance.computational_view(["free_flow_time"])
    zone_ids = np.asarray(impedance.index, dtype=np.int64)

    stops = gpd.read_parquet(CACHE_DIR / "phase1_real" / "stops_demand.parquet")
    stops = stops.reset_index(drop=True)
    centroid_ids = 9_000_000_000 + np.arange(len(stops), dtype=np.int64)
    vectors = pd.DataFrame(
        {
            "productions": stops["population"].to_numpy(dtype=float),
            "attractions": stops["jobs"].to_numpy(dtype=float),
        },
        index=centroid_ids,
    ).reindex(zone_ids)

    if vectors.isna().any().any():
        missing = vectors.index[vectors["productions"].isna()].tolist()[:10]
        raise AequilibraEPipelineError(f"Demand centroids missing from skim matrix: {missing}")

    vectors = vectors.fillna(0.0)
    if vectors["productions"].sum() <= 0 or vectors["attractions"].sum() <= 0:
        raise AequilibraEPipelineError("Productions and attractions must both be positive")
    vectors["attractions"] *= vectors["productions"].sum() / vectors["attractions"].sum()

    model = SyntheticGravityModel()
    model.function = "EXPO"
    model.beta = EXPONENTIAL_BETA

    gravity = GravityApplication(
        impedance=impedance,
        vectors=vectors,
        row_field="productions",
        model=model,
        column_field="attractions",
        output_core="gravity",
        nan_as_zero=True,
    )
    gravity.apply()
    gravity.save_to_project("demand_gravity", "aem", project=project)
    return gravity.output, vectors


def run_assignment(project, demand_matrix) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run BPR/BFW traffic assignment and return link-level results + convergence."""
    from aequilibrae.paths import TrafficAssignment, TrafficClass

    demand_matrix.computational_view(["gravity"])
    graph = project.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(True)

    tc = TrafficClass("demand", graph, demand_matrix)
    assignment = TrafficAssignment()
    assignment.add_class(tc)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.set_algorithm("bfw")
    assignment.max_iter = 1000
    assignment.rgap_target = 0.001
    assignment.execute()

    load = assignment.results().reset_index()
    convergence = assignment.report()
    assignment.save_results("aequilibrae_assignment", keep_zero_flows=False, project=project)
    return load, convergence


def run_all(force: bool = False) -> dict:
    """Run the complete AequilibraE workflow and persist model outputs."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    project_path = build_project(force=force)
    project = _open_project()
    try:
        skim = run_skimming(project)
        demand, vectors = run_gravity(project, skim)
        load, convergence = run_assignment(project, demand)

        out_dir = AEQ_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        load.to_parquet(out_dir / "link_load.parquet", index=False)
        convergence.to_parquet(out_dir / "convergence.parquet", index=False)

        report = {
            "backend": "AequilibraE 1.7.x",
            "project": str(project_path),
            "n_centroids": int(project.network.count_centroids()),
            "n_nodes": int(project.network.count_nodes()),
            "n_links": int(project.network.count_links()),
            "total_productions": round(float(vectors["productions"].sum()), 1),
            "total_attractions": round(float(vectors["attractions"].sum()), 1),
            "total_demand": round(float(demand.get_matrix("gravity").sum()), 1),
            "assignment_rows": int(len(load)),
            "assignment_columns": list(load.columns),
            "convergence_rows": int(len(convergence)),
            "gravity_function": "EXPO",
            "gravity_beta_per_min": EXPONENTIAL_BETA,
            "reference_decay_radius_km": DECAY_RADIUS_KM,
            "assignment": {"vdf": "BPR", "algorithm": "bfw", "rgap_target": 0.001},
        }
        (out_dir / "aequilibrae_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (REPORT_DIR / "aequilibrae_report.md").write_text(
            "\n".join(
                [
                    "# AequilibraE — транспортная модель",
                    "",
                    f"- Узлов: **{report['n_nodes']:,}**",
                    f"- Рёбер: **{report['n_links']:,}**",
                    f"- Зон-центроидов: **{report['n_centroids']:,}**",
                    f"- Генерация поездок: **{report['total_productions']:,.0f}**",
                    f"- Притяжение: **{report['total_attractions']:,.0f}**",
                    f"- Матрица корреспонденций: **{report['total_demand']:,.0f}**",
                    "",
                    "## Распределение",
                    f"- Функция: EXPO, β = {report['gravity_beta_per_min']:.6f} 1/мин",
                    f"- Эквивалентный радиус затухания: {report['reference_decay_radius_km']} км при {REFERENCE_SPEED_KMH} км/ч",
                    "",
                    "## Назначение",
                    "- VDF: BPR",
                    "- Алгоритм: BFW",
                    "- Целевой относительный разрыв: 0.001",
                    "",
                    "AequilibraE project: `data/cache/aequilibrae/project`",
                ]
            ),
            encoding="utf-8",
        )
        return report
    finally:
        project.close()


if __name__ == "__main__":
    print(json.dumps(run_all(force=True), indent=2, ensure_ascii=False))
