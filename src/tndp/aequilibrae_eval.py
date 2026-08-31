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

EVALUATOR_VERSION = "aeq-transit-v10-assignment-segment-load-fleet"

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
            total += float(cached[2])
            continue
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
    if not zones_path.exists():
        return {}
    zones = gpd.read_parquet(zones_path).to_crs("EPSG:4326").reset_index(drop=True)
    cent = zones.geometry.centroid
    zone_xy = np.column_stack([cent.x.to_numpy(float), cent.y.to_numpy(float)])
    stops = np.asarray(stop_xy_lonlat, dtype=float)
    if not len(zone_xy) or not len(stops):
        return {}
    _, idx = cKDTree(stops).query(zone_xy, k=1)
    return {int(stop): int(zone) for zone, stop in enumerate(np.asarray(idx, dtype=int))}

def _adapt_fleet(route_set: RouteSet, demand: np.ndarray, stop_to_zone: dict[int, int], lengths: list[float], config: NetworkDesignConfig):
    current = route_set
    details: list[dict] = []
    for _ in range(3):
        loads = reconstruct_route_loads(current, demand, stop_to_zone=stop_to_zone, route_lengths_km=lengths,
                                        frequencies_vph=[r.frequency_vph for r in current.routes])
        updated, details = [], []
        changed = False
        for i, route in enumerate(current.routes):
            flow = max(loads[i].max_section_flow_pph if i < len(loads) else 0.0, _route_flow(route))
            code, op = select_vehicle_for_route(max_section_flow_pph=flow, route_length_km=max(.001, lengths[i] * 2),
                allowed_vehicle_types=config.allowed_vehicle_types, speed_kmh=config.speed_kmh,
                interval_reserve_sec=config.interval_reserve_sec, terminal_delay_reserve=config.terminal_delay_reserve,
                charging_min_per_terminal=config.charging_min_per_terminal, annual_days=config.annual_days,
                park_trip_coefficient=config.park_trip_coefficient, frequency_profile=config.frequency_profile)
            nr = route.with_flow(flow).with_vehicle_type(code).with_frequency(float(op["frequency_vph"]))
            changed |= nr != route
            updated.append(nr)
            details.append({"max_section_flow_pph": flow, "max_section_index": loads[i].max_section_index,
                            "assigned_demand": loads[i].assigned_demand, **op})
        current = RouteSet(updated)
        if not changed:
            break
    return current, details

def _extract_assignment_segment_loads(graph, transit_class, assignment, adapted: RouteSet, route_lengths: list[float]) -> list[dict]:
    """Extract actual transit assignment results where AequilibraE exposes them.

    AequilibraE skims provide OD-level boardings/transfers, while path loading
    is represented by the transit graph. We use graph link flows when exposed;
    otherwise we explicitly mark the fallback instead of pretending the
    reconstructed OD load is an AequilibraE segment flow.
    """
    candidates = []
    for obj in (getattr(transit_class, "graph", None), getattr(assignment, "results", None), assignment):
        if obj is None:
            continue
        for name in ("link_loads", "loads", "flow", "flows", "transit_link_loads"):
            value = getattr(obj, name, None)
            if value is not None:
                try:
                    arr = np.asarray(value, dtype=float)
                    if arr.size:
                        candidates.append((name, arr))
                except Exception:
                    pass
    if candidates:
        name, arr = max(candidates, key=lambda x: x[1].size)
        return [{"source": "AequilibraE", "field": name, "values": arr.tolist()}]
    return [{"source": "reconstruction_fallback", "values": [], "reason": "AequilibraE public API does not expose route-level segment flow directly"}]

def evaluate_route_set_aequilibrae(route_set: RouteSet, demand: np.ndarray, stop_xy_lonlat: np.ndarray,
                                   project_path: str | Path, config: NetworkDesignConfig, *, road_graph: nx.Graph,
                                   stop_mapping, path_index=None, stop_to_zone: dict[int, int] | None = None,
                                   cache_dir: str | Path | None = None) -> Evaluation:
    total = float(np.asarray(demand, dtype=float).sum())
    if not route_set.routes:
        return Evaluation(score=total * config.uncovered_demand_weight, uncovered_demand=total, metadata={"empty_network": True})
    lengths = [_route_length_km(r, road_graph, stop_mapping, path_index) for r in route_set.routes]
    stop_to_zone = stop_to_zone or _infer_stop_to_zone(stop_xy_lonlat)
    adapted, fleet_details = _adapt_fleet(route_set, demand, stop_to_zone, lengths, config) if stop_to_zone else (route_set, [])
    key = _route_set_key(adapted)
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
    temp_root = Path(tempfile.mkdtemp(prefix="tranmodel_tndp_project_"))
    project = None
    try:
        project_dir = temp_root / "project"
        shutil.copytree(Path(project_path), project_dir, dirs_exist_ok=True)
        public_db = project_dir / "public_transport.sqlite"
        if public_db.exists(): public_db.unlink()
        gtfs = build_gtfs_from_route_set(adapted, stop_xy_lonlat, temp_root / "routes.zip", road_graph=road_graph, stop_mapping=stop_mapping, path_index=path_index)
        project = Project.from_path(project_dir)
        transit = Transit(project)
        builder = transit.new_gtfs_builder(agency="TRANMODEL", file_path=str(gtfs), day="", description="TNDP candidate route set")
        builder.set_allow_map_match(False); builder.load_date("2026-01-15"); builder.execute_import()
        graph_builder = transit.create_graph(projected_crs=f"EPSG:{PROJ_EPSG}", with_inner_stop_transfers=True,
            with_outer_stop_transfers=False, with_walking_edges=True, distance_upper_bound=800.0,
            blocking_centroid_flows=True, connector_method="nearest_neighbour", max_connectors_per_zone=3)
        graph_builder.create_line_geometry(method="connector project match", graph="c"); graph_builder.save()
        graph = graph_builder.to_transit_graph()
        centroids = np.asarray(graph.centroids, dtype=np.int64)
        if len(centroids) != demand.shape[0]:
            raise AequilibraEEvaluationError(f"Transit graph has {len(centroids)} zone centroids but demand has {demand.shape[0]} zones")
        matrix = AequilibraeMatrix(); matrix.create_empty(zones=len(centroids), matrix_names=["pt"], memory_only=True)
        matrix.index = centroids; matrix.matrices[:, :, 0] = demand; matrix.computational_view(["pt"])
        transit_class = TransitClass(name="pt", graph=graph, matrix=matrix)
        assignment = TransitAssignment(); assignment.add_class(transit_class); assignment.set_time_field("trav_time"); assignment.set_frequency_field("freq")
        assignment.set_skimming_fields(["trav_time", "on_board_trav_time", "walking_trav_time", "waiting_time", "transfer_time", "boardings", "transfers"])
        assignment.set_algorithm("os"); transit_class.set_demand_matrix_core("pt"); assignment.execute()
        skim = assignment.get_skim_results()["pt"].matrix
        generalized = np.zeros_like(demand, dtype=float)
        for field, weight in (("trav_time", config.in_vehicle_weight), ("waiting_time", config.wait_weight), ("walking_trav_time", config.walk_weight)):
            if field in skim: generalized += np.nan_to_num(np.asarray(skim[field]), nan=np.inf) * weight
        if "transfers" in skim: generalized += np.nan_to_num(np.asarray(skim["transfers"]), nan=np.inf) * config.transfer_penalty_min * config.transfer_weight
        finite = np.isfinite(generalized); served = float(demand[finite].sum()); uncovered = float(demand[~finite].sum())
        weighted_user_cost = float(np.nansum(demand[finite] * generalized[finite]) / max(served, 1.0))
        transfers_arr = np.nan_to_num(np.asarray(skim.get("transfers", np.zeros_like(demand))), nan=0.0)
        avg_transfers = float(np.nansum(demand * transfers_arr) / max(total, 1.0))
        direct_share = float(demand[(finite) & (transfers_arr == 0)].sum() / max(total, 1.0))
        actual_assignment_loads = _extract_assignment_segment_loads(graph, transit_class, assignment, adapted, lengths)
        # Until a stable public route-level flow API is available, use reconstructed
        # route loads only for fleet sizing and clearly report their provenance.
        reconstructed = reconstruct_route_loads(adapted, demand, stop_to_zone=stop_to_zone, route_lengths_km=lengths,
                                                frequencies_vph=[r.frequency_vph for r in adapted.routes]) if stop_to_zone else []
        route_characteristics=[]; annual_mileage=annual_hours=annual_contract=annual_amortization=0.0; fleet=0
        for i, route in enumerate(adapted.routes):
            one_way=lengths[i]; flow=fleet_details[i]["max_section_flow_pph"] if i < len(fleet_details) else _route_flow(route)
            operating=calculate_route_characteristics(2*one_way, flow, capacity_at_4_ppm2=route.capacity, speed_kmh=config.speed_kmh,
                interval_reserve_sec=config.interval_reserve_sec, terminal_delay_reserve=config.terminal_delay_reserve,
                charging_min_per_terminal=config.charging_min_per_terminal, technical_readiness=route.technical_readiness,
                frequency_profile=config.frequency_profile)
            annual_mileage+=operating.annual_mileage_km; annual_hours+=operating.annual_in_service_hours; fleet+=operating.fleet
            d=fleet_details[i] if i<len(fleet_details) else {}
            annual_contract+=float(d.get("annual_fleet_contract_cost_mln",0)); annual_amortization+=float(d.get("annual_fleet_amortization_mln",0))
            route_characteristics.append({"route_id":route.route_id,"one_way_length_km":one_way,"round_trip_length_km":operating.route_length_km,
                "max_section_flow_pph":flow,"frequency_vph":operating.frequency_vph,"interval_min":operating.interval_min,
                "turnaround_min":operating.turnaround_min,"release":operating.release,"technical_readiness":operating.technical_readiness,
                "fleet":operating.fleet,"daily_trips":operating.daily_trips,"annual_mileage_km":operating.annual_mileage_km,
                "annual_in_service_hours":operating.annual_in_service_hours,"vehicle_type":getattr(route,"vehicle_type","unknown"),
                "vehicle_name":d.get("vehicle_name",""),"capacity":route.capacity,"annual_contract_cost_mln":d.get("annual_fleet_contract_cost_mln",0),
                "annual_amortization_mln":d.get("annual_fleet_amortization_mln",0),"max_section_index":d.get("max_section_index",-1),
                "assigned_demand":d.get("assigned_demand",0),"load_source":"reconstructed_until_route_flow_api"})
        score=weighted_user_cost+annual_mileage*config.operator_route_km_weight+uncovered*config.uncovered_demand_weight+avg_transfers*config.transfer_penalty_min*config.transfer_weight+(annual_contract+annual_amortization)*0.05
        evaluation=Evaluation(score=float(score),user_cost=weighted_user_cost,operator_cost=float(annual_mileage),uncovered_demand=uncovered,transfers=avg_transfers,direct_demand_share=direct_share,
            metadata={"evaluator":"AequilibraE","served_demand":served,"annual_mileage_km":annual_mileage,"annual_in_service_hours":annual_hours,"fleet":fleet,
                      "annual_contract_cost_mln":annual_contract,"annual_amortization_mln":annual_amortization,"route_characteristics":route_characteristics,
                      "assignment_segment_loads":actual_assignment_loads,"segment_load_source":"AequilibraE_graph_if_exposed_else_reconstruction"})
        result_path.write_text(_evaluation_json(evaluation),encoding="utf-8")
        return evaluation
    finally:
        if project is not None:
            try: project.close()
            except Exception: pass
        shutil.rmtree(temp_root,ignore_errors=True)
