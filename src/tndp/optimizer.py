"""Whole-route TNDP optimizer with constructive search and local mutations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

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
    """Select whole routes, then improve them with local route mutations."""

    def __init__(self, candidates: list[Route], evaluator: Evaluator,
                 config: NetworkDesignConfig | None = None) -> None:
        self.candidates = candidates
        self.evaluator = evaluator
        self.config = config or NetworkDesignConfig()
        self.config.validate()

    @staticmethod
    def _sorted_candidates(candidates: list[Route]) -> list[Route]:
        return sorted(candidates, key=lambda r: (r.route_id or "", r.nodes))

    def solve(self, initial: RouteSet | None = None) -> TNDPResult:
        network = initial.copy() if initial else RouteSet()
        current = self.evaluator(network)
        history = [{"phase": "start", "routes": network.route_count(), "score": current.score}]

        remaining = self._sorted_candidates(self.candidates)
        for iteration in range(self.config.iterations):
            if network.route_count() >= self.config.max_routes:
                break

            best_route = None
            best_eval = None
            for route in remaining:
                if network.contains_nodes(route.nodes):
                    continue
                trial = network.copy()
                trial.add(route)
                ev = self.evaluator(trial)
                if best_eval is None or ev.score < best_eval.score:
                    best_route, best_eval = route, ev

            if best_route is None or best_eval is None:
                break

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
        for round_no in range(1, self.config.local_search_rounds + 1):
            best_network = network
            best_eval = current
            best_meta = None
            for trial, index, replacement in mutate_route_set(network, self._graph, self.config):
                ev = self.evaluator(trial)
                if ev.score + self.config.improvement_epsilon < best_eval.score:
                    best_network, best_eval = trial, ev
                    best_meta = (index, replacement)
            if best_meta is None:
                break
            old = network.routes[best_meta[0]]
            network, current = best_network, best_eval
            history.append({
                "phase": "local_search",
                "round": round_no,
                "routes": network.route_count(),
                "score": current.score,
                "replaced_route": list(old.nodes),
                "new_route": list(best_meta[1].nodes),
            })
        return network, current

    def solve_with_graph(self, graph, initial: RouteSet | None = None) -> TNDPResult:
        """Solve while making the candidate road graph available to mutations."""
        self._graph = graph
        return self.solve(initial)


def surrogate_evaluator(
    demand,
    node_xy_km,
    route_set: RouteSet,
    config: NetworkDesignConfig | None = None,
) -> Evaluation:
    """Fast screening evaluator based on direct demand, length and overlap."""
    config = config or NetworkDesignConfig()
    import numpy as np

    matrix = np.asarray(demand, dtype=float)
    served = np.zeros_like(matrix, dtype=bool)
    route_km = 0.0
    covered_pairs = 0
    for route in route_set.routes:
        seq = route.nodes
        node_set = set(seq)
        route_km += float(sum(np.linalg.norm(node_xy_km[a] - node_xy_km[b]) for a, b in zip(seq[:-1], seq[1:])))
        nodes = list(node_set)
        if nodes:
            served[np.ix_(nodes, nodes)] = True
    np.fill_diagonal(served, False)

    total = float(matrix.sum())
    direct = float(matrix[served].sum()) if total else 0.0
    uncovered = max(0.0, total - direct)
    overlap = sum(max(0, sum(1 for r in route_set.routes if node in r.nodes) - 1) for route in route_set.routes for node in set(route.nodes))
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
