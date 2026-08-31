"""Whole-route TNDP optimizer with construction and local network search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model import Evaluation, NetworkDesignConfig, Route, RouteSet
from .mutations import mutate_route_set


@dataclass
class TNDPResult:
    """Result of TNDP construction and local network search."""

    routes: RouteSet
    evaluation: Evaluation
    history: list[dict]


Evaluator = Callable[[RouteSet], Evaluation]
Progress = Callable[[str], None]


class TNDPOptimizer:
    """Construct routes with a cheap screen and spend AequilibraE calls selectively."""

    def __init__(self, candidates: list[Route], evaluator: Evaluator,
                 config: NetworkDesignConfig | None = None,
                 fast_evaluator: Evaluator | None = None,
                 progress: Progress | None = None) -> None:
        self.candidates = candidates
        self.evaluator = evaluator
        self.fast_evaluator = fast_evaluator or evaluator
        self.config = config or NetworkDesignConfig()
        self.config.validate()
        self.progress = progress
        self._graph = None
        self._fast_cache: dict[tuple, Evaluation] = {}
        self._full_cache: dict[tuple, Evaluation] = {}

    @staticmethod
    def _key(network: RouteSet) -> tuple:
        return tuple((r.nodes, round(float(r.frequency_vph), 6)) for r in network.routes)

    def _evaluate(self, network: RouteSet, full: bool = True) -> Evaluation:
        cache = self._full_cache if full else self._fast_cache
        key = self._key(network)
        cached = cache.get(key)
        if cached is not None:
            return cached
        evaluator = self.evaluator if full else self.fast_evaluator
        result = evaluator(network)
        cache[key] = result
        return result

    def _notify(self, message: str) -> None:
        if self.progress:
            self.progress(message)
        print(f"[TNDP] {message}", flush=True)

    def _rank_additions(self, network: RouteSet, remaining: list[Route]) -> list[tuple[float, RouteSet, Route]]:
        ranked: list[tuple[float, RouteSet, Route]] = []
        total = len(remaining)
        for idx, route in enumerate(remaining, 1):
            if network.contains_nodes(route.nodes):
                continue
            trial = network.copy()
            trial.add(route)
            ev = self._evaluate(trial, full=False)
            ranked.append((ev.score, trial, route))
            if idx % 10 == 0:
                self._notify(f"  быстрый отбор: {idx}/{total}")
        ranked.sort(key=lambda x: x[0])
        return ranked

    def solve(self, initial: RouteSet | None = None, graph=None) -> TNDPResult:
        self._graph = graph
        network = initial.copy() if initial else RouteSet()
        current = self._evaluate(network, full=True)
        history = [{"phase": "start", "routes": network.route_count(), "score": current.score}]
        self._notify(f"Старт: {len(self.candidates)} кандидатов, целевая сеть {self.config.min_routes}–{self.config.max_routes} маршрутов")

        remaining = [r for r in self.candidates if not network.contains_nodes(r.nodes)]
        for iteration in range(self.config.iterations):
            if network.route_count() >= self.config.max_routes:
                break

            self._notify(f"Построение: итерация {iteration + 1}/{self.config.iterations}, сеть {network.route_count()} маршрутов")
            ranked = self._rank_additions(network, remaining)
            if not ranked:
                break

            # Evaluate only the best few surrogate candidates with AequilibraE.
            # This changes the expensive-call budget from O(all candidates) to O(k).
            top_k = min(self.config.full_candidates_per_iteration, len(ranked))
            best_route: Route | None = None
            best_network: RouteSet | None = None
            best_eval: Evaluation | None = None
            for rank, (_, trial, route) in enumerate(ranked[:top_k], 1):
                self._notify(f"  точная оценка {rank}/{top_k}: {route.route_id or 'route'}")
                ev = self._evaluate(trial, full=True)
                if best_eval is None or ev.score < best_eval.score:
                    best_route, best_network, best_eval = route, trial, ev

            if best_route is None or best_network is None or best_eval is None:
                break

            must_fill_min = network.route_count() < self.config.min_routes
            if not must_fill_min and best_eval.score + self.config.improvement_epsilon >= current.score:
                self._notify("  улучшения по полной оценке больше нет — завершаем построение")
                break

            network = best_network
            current = best_eval
            remaining = [r for r in remaining if r is not best_route]
            history.append({
                "phase": "construct", "iteration": iteration + 1,
                "routes": network.route_count(), "score": current.score,
                "added_route": list(best_route.nodes), "frequency_vph": best_route.frequency_vph,
            })
            self._notify(f"  выбран маршрут {network.route_count()}: оценка AequilibraE = {current.score:.3f}")

        network, current = self._local_search(network, current, history)
        self._notify(f"Завершено: {network.route_count()} маршрутов, оценка {current.score:.3f}")
        return TNDPResult(network, current, history)

    def _local_search(self, network: RouteSet, current: Evaluation, history: list[dict]):
        if self._graph is None or network.route_count() == 0:
            return network, current

        for round_no in range(1, self.config.local_search_rounds + 1):
            self._notify(f"Локальный поиск: раунд {round_no}/{self.config.local_search_rounds}")
            ranked: list[tuple[float, RouteSet, dict]] = []
            trials = list(mutate_route_set(network, self._graph, self.config))
            total = len(trials)
            for idx, (trial, meta) in enumerate(trials, 1):
                ev = self._evaluate(trial, full=False)
                ranked.append((ev.score, trial, meta))
                if idx % 10 == 0:
                    self._notify(f"  быстрый отбор мутаций: {idx}/{total}")
            ranked.sort(key=lambda x: x[0])
            if not ranked:
                break

            top_k = min(self.config.full_candidates_per_iteration, len(ranked))
            best_network = network
            best_meta = None
            best_full = current
            for rank, (_, trial, meta) in enumerate(ranked[:top_k], 1):
                self._notify(f"  точная оценка мутации {rank}/{top_k}")
                ev = self._evaluate(trial, full=True)
                if ev.score + self.config.improvement_epsilon < best_full.score:
                    best_network, best_full, best_meta = trial, ev, meta

            if best_meta is None:
                break

            network, current = best_network, best_full
            history.append({
                "phase": "local_search", "round": round_no,
                "routes": network.route_count(), "score": current.score,
                "operation": best_meta["operation"], "index": best_meta["index"],
                "new_route": list(best_meta["route"].nodes),
            })
            self._notify(f"  улучшение: {current.score:.3f}")

        return network, current


def surrogate_evaluator(demand, node_xy_km, route_set: RouteSet,
                        config: NetworkDesignConfig | None = None, *args, **kwargs) -> Evaluation:
    """Fast candidate-screening evaluator; final evaluation belongs to AequilibraE."""
    config = config or NetworkDesignConfig()
    matrix = np.asarray(demand, dtype=float)
    total = float(matrix.sum())
    if route_set.route_count() == 0:
        return Evaluation(score=total * config.uncovered_demand_weight if total else 0.0,
                          uncovered_demand=total,
                          metadata={"evaluator": "surrogate", "empty_network": True})
    served = np.zeros_like(matrix, dtype=bool)
    route_km = 0.0
    for route in route_set.routes:
        seq = route.nodes
        route_km += float(sum(np.linalg.norm(node_xy_km[a] - node_xy_km[b]) for a, b in zip(seq[:-1], seq[1:])))
        nodes = list(set(seq))
        if nodes:
            served[np.ix_(nodes, nodes)] = True
    np.fill_diagonal(served, False)
    direct = float(matrix[served].sum()) if total else 0.0
    uncovered = max(0.0, total - direct)
    overlap = sum(max(0, sum(1 for r in route_set.routes if node in r.nodes) - 1)
                  for node in {n for r in route_set.routes for n in r.nodes})
    score = uncovered * config.uncovered_demand_weight + route_km * config.operator_route_km_weight + overlap * config.duplication_weight
    return Evaluation(score=score, operator_cost=route_km, uncovered_demand=uncovered,
                      direct_demand_share=direct / total if total else 0.0,
                      metadata={"evaluator": "surrogate"})
