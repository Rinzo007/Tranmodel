from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from config import CACHE_DIR, PROJ_EPSG
from .gtfs import build_gtfs_from_route_set
from .model import Evaluation, NetworkDesignConfig, RouteSet
from .route_economics import calculate_route_characteristics
from .route_loads import reconstruct_route_loads, select_vehicle_for_route
from .transit_loads import extract_transit_segment_loads
from .vehicle_types import get_vehicle_type

EVALUATOR_VERSION = "aeq-transit-v13-service-plan"
MAX_ASSIGNMENT_FLEET_ITERATIONS = 3

class AequilibraEEvaluationError(RuntimeError):
    pass

def _route_frequency(route: Any) -> float:
    return float(getattr(route, "frequency_vph", getattr(route, "frequency", 6.0)))

def _route_flow(route: Any) -> float:
    return max(float(getattr(route, "max_section_flow_pph", 0.0)), 0.0)

def _route_length_km(route: Any, road_graph: nx.Graph, stop_mapping, path_index=None) -> float:
    total = 0.0
    for a, b in zip(route.nodes[:-1], route.nodes[1:]):
        cached = path_index.get(int(a), int(b)) if path_index is not None else None
        if cached is not None:
            total += float(cached[2]); continue
        path = nx.shortest_path(road_graph, stop_mapping[int(a)], stop_mapping[int(b)], weight="time")
        total += float(nx.path_weight(road_graph, path, weight="length_km"))
    return total

def _route_set_key(route_set: RouteSet) -> str:
    payload = {"version": EVALUATOR_VERSION, "routes": [
        {"nodes": list(r.nodes), "frequency_vph": _route_frequency(r),
         "max_section_flow_pph": _route_flow(r), "vehicle_type": getattr(r, "vehicle_type", "bus")}
        for r in route_set.routes]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _evaluation_json(value: Evaluation) -> str:
    return json.dumps(asdict(value), ensure_ascii=False, default=float, sort_keys=True)

def _infer_stop_to_zone(stop_xy_lonlat: np.ndarray) -> dict[int, int]:
    zones_path = CACHE_DIR / "zones" / "zones.parquet"
    if not zones_path.exists(): return {}
    zones = gpd.read_parquet(zones_path).to_crs("EPSG:4326").reset_index(drop=True)
    cent = zones.geometry.centroid
    zone_xy = np.column_stack([cent.x.to_numpy(float), cent.y.to_numpy(float)])
    stops = np.asarray(stop_xy_lonlat, dtype=float)
    if not len(zone_xy) or not len(stops): return {}
    _, idx = cKDTree(zone_xy).query(stops, k=1)
    return {int(stop): int(zone) for stop, zone in enumerate(np.asarray(idx, dtype=int))}

def _adapt_fleet(route_set: RouteSet, demand: np.ndarray, stop_to_zone: dict[int, int], lengths: list[float], config: NetworkDesignConfig):
    current = route_set; details: list[dict] = []
    for _ in range(2):
        loads = reconstruct_route_loads(current, demand, stop_to_zone=stop_to_zone, route_lengths_km=lengths,
                                        frequencies_vph=[r.frequency_vph for r in current.routes])
        updated, details = [], []; changed = False
        for i, route in enumerate(current.routes):
            flow = max(loads[i].max_section_flow_pph if i < len(loads) else 0.0, _route_flow(route))
            code, op = select_vehicle_for_route(max_section_flow_pph=flow, route_length_km=max(.001, lengths[i] * 2),
                allowed_vehicle_types=config.allowed_vehicle_types, speed_kmh=config.speed_kmh,
                interval_reserve_sec=config.interval_reserve_sec, terminal_delay_reserve=config.terminal_delay_reserve,
                charging_min_per_terminal=config.charging_min_per_terminal, annual_days=config.annual_days,
                park_trip_coefficient=config.park_trip_coefficient, frequency_profile=config.frequency_profile)
            nr = route.with_flow(flow).with_vehicle_type(code).with_frequency(float(op["frequency_vph"]))
            changed |= nr != route; updated.append(nr)
            details.append({"max_section_flow_pph": flow, "max_section_index": loads[i].max_section_index,
                            "assigned_demand": loads[i].assigned_demand, **op})
        current = RouteSet(updated)
        if not changed: break
    return current, details

def _routes_changed(before: RouteSet, after: RouteSet, frequency_tolerance: float = 0.01) -> bool:
    if before.route_count() != after.route_count(): return True
    return any(tuple(a.nodes) != tuple(b.nodes) or getattr(a, "vehicle_type", "bus") != getattr(b, "vehicle_type", "bus") or
               abs(_route_frequency(a) - _route_frequency(b)) > frequency_tolerance
               for a, b in zip(before.routes, after.routes))

def evaluate_route_set_aequilibrae(route_set: RouteSet, demand: np.ndarray, stop_xy_lonlat: np.ndarray,
    project_path: str | Path, config: NetworkDesignConfig, *, road_graph: nx.Graph, stop_mapping,
    path_index=None, stop_to_zone: dict[int, int] | None = None, cache_dir: str | Path | None = None,
    assignment_iteration: int = 0) -> Evaluation:
    """Evaluate a route set with AequilibraE and converge vehicle/frequency service plan."""
    total = float(np.asarray(demand, dtype=float).sum())
    if not route_set.routes:
        return Evaluation(score=total * config.uncovered_demand_weight, uncovered_demand=total, metadata={"empty_network": True})
    lengths = [_route_length_km(r, road_graph, stop_mapping, path_index) for r in route_set.routes]
    stop_to_zone = stop_to_zone or _infer_stop_to_zone(stop_xy_lonlat)
    adapted, fleet_details = _adapt_fleet(route_set, demand, stop_to_zone, lengths, config) if stop_to_zone else (route_set, [])
    key = _route_set_key(adapted)
    root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="tranmodel_tndp_eval_")); root.mkdir(parents=True, exist_ok=True)
    result_path = root / f"{key}.json"
    if result_path.exists(): return Evaluation(**json.loads(result_path.read_text(encoding="utf-8")))
    try:
        from aequilibrae import Project
        from aequilibrae.matrix import AequilibraeMatrix
        from aequilibrae.paths import TransitAssignment, TransitClass
        from aequilibrae.transit import Transit
    except ImportError as exc:
        raise AequilibraEEvaluationError("AequilibraE is required for full TNDP evaluation") from exc
    temp_root = Path(tempfile.mkdtemp(prefix="tranmodel_tndp_project_")); project = None
    try:
        project_dir = temp_root / "project"; shutil.copytree(Path(project_path), project_dir, dirs_exist_ok=True)
        public_db = project_dir / "public_transport.sqlite"
        if public_db.exists(): public_db.unlink()
        gtfs = build_gtfs_from_route_set(adapted, stop_xy_lonlat, temp_root / "routes.zip", road_graph=road_graph, stop_mapping=stop_mapping, path_index=path_index)
        project = Project.from_path(project_dir); transit = Transit(project)
        builder = transit.new_gtfs_builder(agency="TRANMODEL", file_path=str(gtfs), day="", description="TNDP candidate route set")
        builder.set_allow_map_match(False); builder.load_date("2026-01-15"); builder.execute_import()
        graph_builder = transit.create_graph(projected_crs=f"EPSG:{PROJ_EPSG}", with_inner_stop_transfers=True,
            with_outer_stop_transfers=False, with_walking_edges=True, distance_upper_bound=800.0,
            blocking_centroid_flows=True, connector_method="nearest_neighbour", max_connectors_per_zone=3)
        graph_builder.create_line_geometry(method="connector project match", graph="c"); graph_builder.save(); graph = graph_builder.to_transit_graph()
        centroids = np.asarray(graph.centroids, dtype=np.int64)
        if len(centroids) != demand.shape[0]:
            raise AequilibraEEvaluationError(f"Transit graph has {len(centroids)} zone centroids but demand has {demand.shape[0]} zones")
        matrix = AequilibraeMatrix(); matrix.create_empty(zones=len(centroids), matrix_names=["pt"], memory_only=True)
        matrix.index = centroids; matrix.matrices[:, :, 0] = demand; matrix.computational_view(["pt"])
        transit_class = TransitClass(name="pt", graph=graph, matrix=matrix); assignment = TransitAssignment(); assignment.add_class(transit_class)
        assignment.set_time_field("trav_time"); assignment.set_frequency_field("freq")
        assignment.set_skimming_fields(["trav_time", "on_board_trav_time", "walking_trav_time", "waiting_time", "transfer_time", "boardings", "transfers"])
        assignment.set_algorithm("os"); transit_class.set_demand_matrix_core("pt"); assignment.execute()
        skim = assignment.get_skim_results()["pt"].matrix
        def field_array(name: str, default: float = 0.0):
            try: return np.asarray(skim[name], dtype=float)
            except Exception: return np.full_like(demand, default, dtype=float)
        ride = np.nan_to_num(field_array("on_board_trav_time", np.inf), nan=np.inf, posinf=np.inf)
        wait = np.nan_to_num(field_array("waiting_time", np.inf), nan=np.inf, posinf=np.inf)
        walk = np.nan_to_num(field_array("walking_trav_time", np.inf), nan=np.inf, posinf=np.inf)
        transfers_arr = np.nan_to_num(field_array("transfers", 0.0), nan=0.0)
        generalized = (config.in_vehicle_weight * ride + config.wait_weight * wait + config.walk_weight * walk +
                       config.transfer_weight * transfers_arr * config.transfer_penalty_min)
        finite = np.isfinite(generalized); served = float(demand[finite].sum()); uncovered = float(demand[~finite].sum())
        weighted_user_cost = float(np.nansum(demand[finite] * generalized[finite]) / max(served, 1.0))
        avg_transfers = float(np.nansum(demand * transfers_arr) / max(total, 1.0))
        direct_share = float(demand[(finite) & (transfers_arr == 0)].sum() / max(total, 1.0))
        exact_loads = extract_transit_segment_loads(project_dir, transit_class.results); exact_max = exact_loads["max_sections"]
        final_routes, final_details = [], []
        for i, route in enumerate(adapted.routes):
            exact = exact_max.get(i); fallback = fleet_details[i] if i < len(fleet_details) else {}
            flow = float(exact["max_section_flow_pph"]) if exact is not None else float(fallback.get("max_section_flow_pph", _route_flow(route)))
            max_idx = int(exact["max_section_index"]) if exact is not None else int(fallback.get("max_section_index", -1))
            load_source = "AequilibraE TransitAssignmentResults" if exact is not None else "reconstructed_fallback"
            code, op = select_vehicle_for_route(max_section_flow_pph=flow, route_length_km=max(.001, lengths[i] * 2),
                allowed_vehicle_types=config.allowed_vehicle_types, speed_kmh=config.speed_kmh, interval_reserve_sec=config.interval_reserve_sec,
                terminal_delay_reserve=config.terminal_delay_reserve, charging_min_per_terminal=config.charging_min_per_terminal,
                annual_days=config.annual_days, park_trip_coefficient=config.park_trip_coefficient, frequency_profile=config.frequency_profile)
            final_routes.append(route.with_flow(flow).with_vehicle_type(code).with_frequency(float(op["frequency_vph"])))
            final_details.append({"max_section_flow_pph": flow, "max_section_index": max_idx, "load_source": load_source, **op})
        assigned_route_set = RouteSet(final_routes)
        if assignment_iteration < MAX_ASSIGNMENT_FLEET_ITERATIONS - 1 and _routes_changed(adapted, assigned_route_set):
            return evaluate_route_set_aequilibrae(assigned_route_set, demand, stop_xy_lonlat, project_path, config,
                road_graph=road_graph, stop_mapping=stop_mapping, path_index=path_index, stop_to_zone=stop_to_zone,
                cache_dir=cache_dir, assignment_iteration=assignment_iteration + 1)
        adapted = assigned_route_set
        annual_mileage = annual_hours = annual_contract = annual_amortization = 0.0; fleet = 0; route_characteristics = []
        for i, route in enumerate(adapted.routes):
            one_way = lengths[i]; d = final_details[i]; vehicle = get_vehicle_type(route.vehicle_type)
            operating = calculate_route_characteristics(2 * one_way, d["max_section_flow_pph"], capacity_at_4_ppm2=route.capacity,
                speed_kmh=config.speed_kmh, interval_reserve_sec=config.interval_reserve_sec, terminal_delay_reserve=config.terminal_delay_reserve,
                charging_min_per_terminal=config.charging_min_per_terminal, charging_at_terminal=vehicle.charging_at_terminal,
                technical_readiness=route.technical_readiness, frequency_profile=config.frequency_profile)
            annual_mileage += operating.annual_mileage_km; annual_hours += operating.annual_in_service_hours; fleet += operating.fleet
            annual_contract += float(d.get("annual_fleet_contract_cost_mln", 0.0)); annual_amortization += float(d.get("annual_fleet_amortization_mln", 0.0))
            route_characteristics.append({"route_id": route.route_id, "one_way_length_km": one_way, "round_trip_length_km": operating.route_length_km,
                "max_section_flow_pph": d["max_section_flow_pph"], "frequency_vph": operating.frequency_vph, "interval_min": operating.interval_min,
                "turnaround_min": operating.turnaround_min, "release": operating.release, "technical_readiness": operating.technical_readiness,
                "fleet": operating.fleet, "daily_trips": operating.daily_trips, "annual_mileage_km": operating.annual_mileage_km,
                "annual_in_service_hours": operating.annual_in_service_hours, "vehicle_type": route.vehicle_type, "vehicle_name": d.get("vehicle_name", ""),
                "capacity": route.capacity, "annual_contract_cost_mln": d.get("annual_fleet_contract_cost_mln", 0.0),
                "annual_amortization_mln": d.get("annual_fleet_amortization_mln", 0.0), "max_section_index": d["max_section_index"], "load_source": d["load_source"]})
        score = weighted_user_cost + annual_mileage * config.operator_route_km_weight + uncovered * config.uncovered_demand_weight +
        score += avg_transfers * config.transfer_penalty_min * config.transfer_weight + (annual_contract + annual_amortization) * 0.05
        evaluation = Evaluation(score=float(score), user_cost=weighted_user_cost, operator_cost=float(annual_mileage), uncovered_demand=uncovered,
            transfers=avg_transfers, direct_demand_share=direct_share,
            metadata={"evaluator": "AequilibraE", "served_demand": served, "annual_mileage_km": annual_mileage,
            "annual_in_service_hours": annual_hours, "fleet": fleet, "annual_contract_cost_mln": annual_contract,
            "annual_amortization_mln": annual_amortization, "route_characteristics": route_characteristics,
            "assignment_segment_loads": exact_loads, "segment_load_source": "AequilibraE TransitAssignmentResults",
            "assignment_iteration": assignment_iteration + 1, "assignment_fleet_converged": True,
            "assignment_fleet_max_iterations": MAX_ASSIGNMENT_FLEET_ITERATIONS})
        result_path.write_text(_evaluation_json(evaluation), encoding="utf-8"); return evaluation
    finally:
        if project is not None:
            try: project.close()
            except Exception: pass
        shutil.rmtree(temp_root, ignore_errors=True)
