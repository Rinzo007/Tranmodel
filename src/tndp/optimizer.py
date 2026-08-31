"""Whole-route TNDP optimizer with surrogate screening and beam search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .model import Evaluation, NetworkDesignConfig, Route, RouteSet
from .mutations import mutate_route_set


@dataclass
class TNDPResult:
    routes: RouteSet
    evaluation: Evaluation
    history: list[dict]


Evaluator = Callable[[RouteSet], Evaluation]
Progress = Callable[[str], None]


class TNDPOptimizer:
    """Search route networks while reserving AequilibraE for a small top-K set."""

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
        # Route order does not affect a transit network, so normalize it.
        return tuple(sorted((r.nodes, round(float(r.frequency_vph), 6)) for r in network.routes))

    def _evaluate(self, network: RouteSet, full: bool = True) -> Evaluation:
        cache = self._full_cache if full else self._fast_cache
        key = self._key(network)
        if key not in cache:
            cache[key] = (self.evaluator if full else self.fast_evaluator)(network)
        return cache[key]

    def _notify(self, message: str) -> None:
        if self.progress:
            self.progress(message)
        print(f"[TNDP] {message}", flush=True)

    def _rank_additions(self, network: RouteSet, remaining: list[Route]) -> list[tuple[float, RouteSet, Route]]:
        ranked = []
        total = len(remaining)
        for idx, route in enumerate(remaining, 1):
            if network.contains_nodes(route.nodes):
                continue
            trial = network.copy()
            trial.add(route)
            ranked.append((self._evaluate(trial, full=False).score, trial, route))
            if idx % 25 == 0:
                self._notify(f"  быстрый отбор: {idx}/{total}")
        ranked.sort(key=lambda x: x[0])
        return ranked

    def solve(self, initial: RouteSet | None = None, graph=None) -> TNDPResult:
        self._graph = graph
        start = initial.copy() if initial else RouteSet()
        current = self._evaluate(start, full=True)
        history = [{"phase": "start", "routes": start.route_count(), "score": current.score}]
        self._notify(f"Старт: {len(self.candidates)} кандидатов, целевая сеть {self.config.min_routes}–{self.config.max_routes} маршрутов")

        # Keep a small beam of complete networks. This avoids committing too early
        # to a locally good first route while keeping expensive evaluation bounded.
        beam: list[tuple[RouteSet, Evaluation]] = [(start, current)]
        for iteration in range(self.config.iterations):
            expanded: list[tuple[float, RouteSet, Evaluation]] = []
            for base, _ in beam:
                if base.route_count() >= self.config.max_routes:
                    expanded.append((self._evaluate(base, full=False).score, base, self._evaluate(base, full=True)))
                    continue
                remaining = [r for r in self.candidates if not base.contains_nodes(r.nodes)]
                ranked = self._rank_additions(base, remaining)
                for score, trial, _ in ranked[:self.config.beam_expansion_per_state]:
                    expanded.append((score, trial, None))
            if not expanded:
                break

            expanded.sort(key=lambda x: x[0])
            unique: dict[tuple, tuple[RouteSet, Evaluation | None]] = {}
            for _, trial, ev in expanded:
                unique.setdefault(self._key(trial), (trial, ev))
                if len(unique) >= self.config.beam_width * 3:
                    break
            states = list(unique.values())
            states.sort(key=lambda x: self._evaluate(x[0], full=False).score)
            states = states[:self.config.beam_width]

            new_beam: list[tuple[RouteSet, Evaluation]] = []
            top_full = states[:min(self.config.full_candidates_per_iteration, len(states))]
            for rank, (trial, cached_ev) in enumerate(top_full, 1):
                self._notify(f"  точная оценка {rank}/{len(top_full)}: сеть из {trial.route_count()} маршрутов")
                ev = cached_ev or self._evaluate(trial, full=True)
                new_beam.append((trial, ev))
            # Keep non-top states alive using their surrogate score until next round.
            for trial, _ in states[len(top_full):]:
                new_beam.append((trial, self._evaluate(trial, full=False)))
            new_beam.sort(key=lambda x: x[1].score)
            beam = new_beam[:self.config.beam_width]
            best_network, best_eval = min(beam, key=lambda x: x[1].score)
            current = self._evaluate(best_network, full=True)
            history.append({"phase": "construct", "iteration": iteration + 1,
                            "routes": best_network.route_count(), "score": current.score,
                            "beam_width": len(beam)})
            self._notify(f"  лучшая сеть: {best_network.route_count()} маршрутов, оценка {current.score:.3f}")
            if best_network.route_count() >= self.config.max_routes:
                break

        network, current = min(beam, key=lambda x: self._evaluate(x[0], full=True))
        network, current = self._local_search(network, current, history)
        self._notify(f"Завершено: {network.route_count()} маршрутов, оценка {current.score:.3f}")
        return TNDPResult(network, current, history)

    def _local_search(self, network: RouteSet, current: Evaluation, history: list[dict]):
        if self._graph is None or network.route_count() == 0:
            return network, current
        for round_no in range(1, self.config.local_search_rounds + 1):
            self._notify(f"Локальный поиск: раунд {round_no}/{self.config.local_search_rounds}")
            trials = list(mutate_route_set(network, self._graph, self.config))
            ranked = sorted((self._evaluate(trial, full=False).score, trial, meta)
                            for trial, meta in trials)
            top_k = min(self.config.full_candidates_per_iteration, len(ranked))
            best_network, best_full, best_meta = network, current, None
            for rank, (_, trial, meta) in enumerate(ranked[:top_k], 1):
                self._notify(f"  точная оценка мутации {rank}/{top_k}")
                ev = self._evaluate(trial, full=True)
                if ev.score + self.config.improvement_epsilon < best_full.score:
                    best_network, best_full, best_meta = trial, ev, meta
            if best_meta is None:
                break
            network, current = best_network, best_full
            history.append({"phase": "local_search", "round": round_no,
                            "routes": network.route_count(), "score": current.score,
                            "operation": best_meta["operation"], "index": best_meta["index"],
                            "new_route": list(best_meta["route"].nodes)})
            self._notify(f"  улучшение: {current.score:.3f}")
        return network, current


def surrogate_evaluator(demand, node_xy_km, route_set: RouteSet,
                        config: NetworkDesignConfig | None = None, *args, **kwargs) -> Evaluation:
    """Fast vectorized coverage evaluator used before exact transit assignment."""
    import numpy as np
    config = config or NetworkDesignConfig()
    matrix = np.asarray(demand, dtype=float)
    total = float(matrix.sum())
    if not route_set.routes:
        return Evaluation(score=total * config.uncovered_demand_weight if total else 0.0,
                          uncovered_demand=total, metadata={"evaluator": "surrogate", "empty_network": True})
    route_km = 0.0
    served = np.zeros(matrix.shape, dtype=bool)
    for route in route_set.routes:
        seq = np.asarray(route.nodes, dtype=int)
        if len(seq) < 2:
            continue
        xy = np.asarray(node_xy_km)[seq]
        route_km += float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum())
        nodes = np.unique(seq)
        if len(nodes):
            served[np.ix_(nodes, nodes)] = True
    np.fill_diagonal(served, False)
    direct = float(matrix[served].sum()) if total else 0.0
    uncovered = max(0.0, total - direct)
    node_counts: dict[int, int] = {}
    for route in route_set.routes:
        for node in set(route.nodes):
            node_counts[node] = node_counts.get(node, 0) + 1
    overlap = sum(max(0, count - 1) for count in node_counts.values())
    score = (uncovered * config.uncovered_demand_weight
             + route_km * config.operator_route_km_weight
             + overlap * config.duplication_weight)
    return Evaluation(score=score, operator_cost=route_km, uncovered_demand=uncovered,
                      direct_demand_share=direct / total if total else 0.0,
                      metadata={"evaluator": "surrogate"})
