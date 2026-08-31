"""Iterative transit network design optimizer.

The optimizer is deliberately assignment-agnostic. It can use a fast
surrogate evaluator during candidate screening and a full AequilibraE
assignment evaluator for the final selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .model import NetworkDesignConfig, Route, RouteSet


@dataclass(frozen=True, slots=True)
class Evaluation:
    score: float
    user_cost: float = 0.0
    operator_cost: float = 0.0
    uncovered_demand: float = 0.0
    transfers: float = 0.0
    direct_demand_share: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class TNDPResult:
    routes: RouteSet
    evaluation: Evaluation
    history: list[dict]


Evaluator = Callable[[RouteSet], Evaluation]


class TNDPOptimizer:
    """Greedy-by-whole-route construction followed by local route mutations.

    This is not the old stop-by-stop greedy route builder. A complete route is
    evaluated against the whole network before it can enter the solution.
    """

    def __init__(self, candidates: list[Route], evaluator: Evaluator,
                 config: NetworkDesignConfig | None = None) -> None:
        self.candidates = candidates
        self.evaluator = evaluator
        self.config = config or NetworkDesignConfig()

    def solve(self, initial: RouteSet | None = None) -> TNDPResult:
        network = initial.copy() if initial else RouteSet()
        current = self.evaluator(network)
        history = [{"iteration": 0, "routes": network.route_count(), "score": current.score}]

        candidates = list(self.candidates)
        for iteration in range(self.config.iterations):
            if network.route_count() >= self.config.max_routes:
                break

            best_route: Route | None = None
            best_eval: Evaluation | None = None
            for route in candidates:
                if network.contains_nodes(route.nodes):
                    continue
                trial = network.copy()
                trial.add(route)
                ev = self.evaluator(trial)
                if best_eval is None or ev.score < best_eval.score:
                    best_route, best_eval = route, ev

            if best_route is None or best_eval is None:
                break
            if best_eval.score + self.config.improvement_epsilon >= current.score:
                break

            network.add(best_route)
            current = best_eval
            history.append({
                "iteration": iteration + 1,
                "routes": network.route_count(),
                "score": current.score,
                "added_route": best_route.nodes,
            })

        network, current = self._local_search(network, current, candidates, history)
        if network.route_count() < self.config.min_routes:
            # Keep the best feasible construction instead of silently returning
            # an underspecified network when enough candidates are available.
            for route in candidates:
                if network.route_count() >= self.config.min_routes:
                    break
                if network.contains_nodes(route.nodes):
                    continue
                trial = network.copy()
                trial.add(route)
                ev = self.evaluator(trial)
                if ev.score < current.score or network.route_count() == 0:
                    network, current = trial, ev
        return TNDPResult(network, current, history)

    def _local_search(self, network: RouteSet, current: Evaluation,
                      candidates: list[Route], history: list[dict]):
        improved = True
        rounds = 0
        while improved and rounds < 10:
            improved = False
            rounds += 1
            for index, old in enumerate(list(network.routes)):
                for replacement in candidates:
                    if replacement == old:
                        continue
                    trial = network.copy()
                    trial.routes[index] = replacement
                    if len(trial.unique_undirected_signatures()) != trial.route_count():
                        continue
                    ev = self.evaluator(trial)
                    if ev.score + self.config.improvement_epsilon < current.score:
                        network, current = trial, ev
                        improved = True
                        history.append({
                            "local_round": rounds,
                            "routes": network.route_count(),
                            "score": current.score,
                            "replaced_route": old.nodes,
                            "new_route": replacement.nodes,
                        })
                        break
                if improved:
                    break
        return network, current


def surrogate_evaluator(
    demand: np.ndarray,
    node_xy_km: np.ndarray,
    route_set: RouteSet,
    config: NetworkDesignConfig | None = None,
) -> Evaluation:
    """Fast evaluator for candidate screening.

    Demand is considered directly served when an OD pair's endpoints are both
    on at least one common directed/undirected route. This deliberately
    underestimates the richness of real transit assignment and is intended for
    pre-screening only.
    """
    config = config or NetworkDesignConfig()
    matrix = np.asarray(demand, dtype=float)
    served = np.zeros_like(matrix, dtype=bool)
    route_km = 0.0
    for route in route_set.routes:
        nodes = set(route.nodes)
        seq = route.nodes
        route_km += float(sum(np.linalg.norm(node_xy_km[a] - node_xy_km[b])
                              for a, b in zip(seq[:-1], seq[1:])))
        for o in nodes:
            served[o, list(nodes)] = True
    np.fill_diagonal(served, False)
    total = float(matrix.sum())
    direct = float(matrix[served].sum()) if total else 0.0
    uncovered = max(0.0, total - direct)
    score = (
        uncovered * config.uncovered_demand_weight
        + route_km * config.operator_route_km_weight
    )
    return Evaluation(
        score=score,
        operator_cost=route_km,
        uncovered_demand=uncovered,
        direct_demand_share=direct / total if total else 0.0,
    )
