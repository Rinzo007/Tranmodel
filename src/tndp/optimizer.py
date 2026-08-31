"""Whole-route TNDP optimizer with construction and local network search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .model import NetworkDesignConfig, Route, RouteSet
from .mutations import mutate_route_set


@dataclass(frozen=True, slots=True)
class Evaluation:
    score: float
    user_cost: float = 0.0
    operator_cost: float = 0.0
    uncovered_demand: float = 0.0
    transfers: float = 0.0
    direct_demand_share: float = 0.0
    capacity_excess: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class TNDPResult:
    routes: RouteSet
    evaluation: Evaluation
    history: list[dict]


Evaluator = Callable[[RouteSet], Evaluation]


class TNDPOptimizer:
    """Construct routes, then improve the complete network with local moves."""

    def __init__(self, candidates: list[Route], evaluator: Evaluator,
                 config: NetworkDesignConfig | None = None) -> None:
        self.candidates = candidates
        self.evaluator = evaluator
        self.config = config or NetworkDesignConfig()
        self.config.validate()
        self._graph = None

    def _evaluate(self, network: RouteSet) -> Evaluation:
        return self.evaluator(network)

    def solve(self, initial: RouteSet | None = None, graph=None) -> TNDPResult:
        self._graph = graph
        network = initial.copy() if initial else RouteSet()
        current = self._evaluate(network)
        history = [{
            "phase": "start",
            "routes": network.route_count(),
            "score": current.score,
        }]

        remaining = [r for r in self.candidates if not network.contains_nodes(r.nodes)]
        for iteration in range(self.config.iterations):
            if network.route_count() >= self.config.max_routes:
                break

            best_route: Route | None = None
            best_eval: Evaluation | None = None
            for route in remaining:
                if network.contains_nodes(route.nodes):
                    continue
                trial = network.copy()
                trial.add(route)
                ev = self._evaluate(trial)
                if best_eval is None or ev.score < best_eval.score:
                    best_route, best_eval = route, ev

            if best_route is None or best_eval is None:
                break

            # Before min_routes the optimizer is required to build a valid-sized
            # network. Afterwards it only accepts a true improvement.
            must_fill_min = network.route_count() < self.config.min_routes
            if not must_fill_min and best_eval.score + self.config.improvement_epsilon >= current.score:
                break

            network.add(best_route)
            current = best_eval
            remaining = [r for r in remaining if r is not best_route]
            history.append({
                "phase": "construct",
                "iteration": iteration + 1,
                "routes": network.route_count(),
                "score": current.score,
                "added_route": list(best_route.nodes),
                "frequency_vph": best_route.frequency_vph,
            })

        network, current = self._local_search(network, current, history)
        return TNDPResult(network, current, history)

    def _local_search(self, network: RouteSet, current: Evaluation, history: list[dict]):
        if self._graph is None or network.route_count() == 0:
            return network, current

        for round_no in range(1, self.config.local_search_rounds + 1):
            best_network = network
            best_eval = current
            best_meta = None

            for trial, meta in mutate_route_set(network, self._graph, self.config):
                ev = self._evaluate(trial)
                if ev.score + self.config.improvement_epsilon < best_eval.score:
                    best_network, best_eval, best_meta = trial, ev, meta

            if best_meta is None:
                break

            network, current = best_network, best_eval
            entry = {
                "phase": "local_search",
                "round": round_no,
                "routes": network.route_count(),
                "score": current.score,
                "operation": best_meta["operation"],
                "index": best_meta["index"],
                "new_route": list(best_meta["route"].nodes),
            }
            history.append(entry)

        return network, current


def surrogate_evaluator(demand, node_xy_km, route_set: RouteSet,
                        config: NetworkDesignConfig | None = None) -> Evaluation:
    """Fast candidate-screening evaluator; final evaluation belongs to AequilibraE."""
    config = config or NetworkDesignConfig()
    matrix = np.asarray(demand, dtype=float)
    total = float(matrix.sum())
    if route_set.route_count() == 0:
        return Evaluation(
            score=total * config.uncovered_demand_weight if total else 0.0,
            uncovered_demand=total,
            metadata={"evaluator": "surrogate", "empty_network": True},
        )

    served = np.zeros_like(matrix, dtype=bool)
    route_km = 0.0
    for route in route_set.routes:
        seq = route.nodes
        route_km += float(sum(np.linalg.norm(node_xy_km[a] - node_xy_km[b])
                              for a, b in zip(seq[:-1], seq[1:])))
        nodes = list(set(seq))
        if nodes:
            served[np.ix_(nodes, nodes)] = True
    np.fill_diagonal(served, False)

    direct = float(matrix[served].sum()) if total else 0.0
    uncovered = max(0.0, total - direct)
    overlap = sum(
        max(0, sum(1 for r in route_set.routes if node in r.nodes) - 1)
        for node in {n for r in route_set.routes for n in r.nodes}
    )
    score = (
        uncovered * config.uncovered_demand_weight
        + route_km * config.operator_route_km_weight
        + overlap * config.duplication_weight
    )
    return Evaluation(
        score=score,
        operator_cost=route_km,
        uncovered_demand=uncovered,
        direct_demand_share=direct / total if total else 0.0,
        metadata={"evaluator": "surrogate"},
    )
