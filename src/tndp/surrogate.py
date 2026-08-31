"""Fast TNDP screening when demand zones and transit stops are separate."""

from __future__ import annotations

import numpy as np

from .model import Evaluation, NetworkDesignConfig, RouteSet


def surrogate_evaluator(demand: np.ndarray, zone_xy_km: np.ndarray,
                        route_set: RouteSet, config: NetworkDesignConfig,
                        zone_to_stop: np.ndarray) -> Evaluation:
    """Screen route sets using zone-to-nearest-stop accessibility.

    This is deliberately a pre-screen only. Final scoring is delegated to
    AequilibraE TransitAssignment.
    """
    matrix = np.asarray(demand, dtype=float)
    total = float(matrix.sum())
    if route_set.route_count() == 0:
        return Evaluation(score=total * config.uncovered_demand_weight,
                          uncovered_demand=total, direct_demand_share=0.0,
                          metadata={"evaluator": "surrogate", "empty_network": True})

    served = np.zeros_like(matrix, dtype=bool)
    route_lengths = 0.0
    for route in route_set.routes:
        nodes = np.asarray(route.nodes, dtype=int)
        if len(nodes) < 2:
            continue
        route_lengths += float(np.linalg.norm(zone_xy_km[zone_to_stop[nodes[:-1]]] -
                                              zone_xy_km[zone_to_stop[nodes[1:]]], axis=1).sum())
        served_stops = set(nodes.tolist())
        reachable = [i for i, stop in enumerate(zone_to_stop) if stop in served_stops]
        if reachable:
            served[np.ix_(reachable, reachable)] = True
    np.fill_diagonal(served, False)
    direct = float(matrix[served].sum())
    uncovered = max(0.0, total - direct)
    overlap = 0
    for i in range(len(route_set.routes)):
        a = set(route_set.routes[i].nodes)
        for j in range(i + 1, len(route_set.routes)):
            overlap += len(a.intersection(route_set.routes[j].nodes))
    score = uncovered * config.uncovered_demand_weight + route_lengths * config.operator_route_km_weight + overlap * config.duplication_weight
    return Evaluation(score=float(score), operator_cost=float(route_lengths),
                      uncovered_demand=uncovered,
                      direct_demand_share=direct / total if total else 0.0,
                      metadata={"evaluator": "zone-stop surrogate"})
