"""Cache-aware wrapper for the complete six-period transit evaluation."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict

from .aequilibrae_eval_cache import load_json, save_json, stable_route_set_key
from .multiperiod_assignment import evaluate_route_set_aequilibrae_periods
from .model import Evaluation


def evaluate_route_set_aequilibrae_periods_cached(route_set, base_demand, stop_xy_lonlat, project_path, config, *, road_graph, stop_mapping, path_index=None, stop_to_zone=None, cache_dir=None, demand_factors=None, progress=None):
    """Cache a complete JSON-serializable six-period Evaluation.

    The cache key includes routes, frequency, vehicle type, configuration and
    the six-period demand/frequency factors. The underlying AequilibraE object
    graph is never persisted.
    """
    if cache_dir is None:
        return evaluate_route_set_aequilibrae_periods(route_set, base_demand, stop_xy_lonlat, project_path, config,
            road_graph=road_graph, stop_mapping=stop_mapping, path_index=path_index, stop_to_zone=stop_to_zone,
            cache_dir=None, demand_factors=demand_factors, progress=progress)
    extra = {"kind": "six_period_evaluation", "project_path": str(project_path), "demand_factors": demand_factors or {}}
    key = stable_route_set_key(route_set, config, extra=extra)
    root = Path(cache_dir) / "six_period"
    cached = load_json(root, key)
    if isinstance(cached, dict) and cached.get("_schema") == 1:
        e = cached.get("evaluation", {})
        return Evaluation(score=float(e.get("score", 0.0)), user_cost=float(e.get("user_cost", 0.0)),
            operator_cost=float(e.get("operator_cost", 0.0)), uncovered_demand=float(e.get("uncovered_demand", 0.0)),
            transfers=float(e.get("transfers", 0.0)), direct_demand_share=float(e.get("direct_demand_share", 0.0)),
            capacity_excess=float(e.get("capacity_excess", 0.0)), metadata=cached.get("metadata", {}))
    ev = evaluate_route_set_aequilibrae_periods(route_set, base_demand, stop_xy_lonlat, project_path, config,
        road_graph=road_graph, stop_mapping=stop_mapping, path_index=path_index, stop_to_zone=stop_to_zone,
        cache_dir=cache_dir, demand_factors=demand_factors, progress=progress)
    save_json(root, key, {"_schema": 1, "evaluation": asdict(ev), "metadata": ev.metadata or {}})
    return ev
