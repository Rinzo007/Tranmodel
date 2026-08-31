"""Whole-route TNDP optimizer with surrogate screening and bounded exact search."""

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
    """Search route networks while reserving AequilibraE for the best states."""

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
        return tuple(sorted((r.nodes, round(float(r.frequency_vph), 6), r.vehicle_type) for r in network.routes))

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

    def _construct_initial_beam(self) -> list[RouteSet]:
        """Construct feasible starting networks with the surrogate only.

        Building to min_routes first avoids spending an exact AequilibraE
        assignment on empty or trivially small networks.
        """
        states = [RouteSet()]
        target = min(self.config.min_routes, self.config.max_routes, len(self.candidates))
        for _ in range(target):
            expanded: list[tuple[float, RouteSet]] = []
            for state in states:
                remaining = [r for r in self.candidates if not state.contains_nodes(r.nodes)]
                ranked = self._rank_additions(state, remaining)
                expanded.extend((score, trial) for score, trial, _ in ranked[:self.config.beam_expansion_per_state])
            if not expanded:
                break
            expanded.sort(key=lambda x: x[0])
            unique: dict[tuple, RouteSet] = {}
            for _, state in expanded:
                unique.setdefault(self._key(state), state)
                if len(unique) >= self.config.beam_width:
                    break
            states = list(unique.values())
        return states

    def solve(self, initial: RouteSet | None = None, graph=None) -> TNDPResult:
        self._graph = graph
        if initial is not None and initial.route_count():
            beam = [initial.copy()]
        else:
            self._notify("Формируем начальные допустимые сети быстрым оценщиком...")
            beam = self._construct_initial_beam()
            if not beam:
                raise RuntimeError("TNDP could not construct an initial route network")

        scored = [(network, self._evaluate(network, full=True)) for network in beam]
        network, current = min(scored, key=lambda x: x[1].score)
        history = [{"phase": "start", "routes": network.route_count(), "score": current.score,
                    "beam_width": len(beam)}]
        self._notify(f"Старт точной оценки: {network.route_count()} маршрутов")

        beam = scored
        for iteration in range(self.config.iterations):
            expanded: list[tuple[float, RouteSet]] = []
            for base, _ in beam:
                if base.route_count() >= self.config.max_routes:
                    expanded.append((self._evaluate(base, full=False).score, base))
                    continue
                remaining = [r for r in self.candidates if not base.contains_nodes(r.nodes)]
                ranked = self._rank_additions(base, remaining)
                expanded.extend((score, trial) for score, trial, _ in ranked[:self.config.beam_expansion_per_state])

            if not expanded:
                break
            expanded.sort(key=lambda x: x[0])
            unique: dict[tuple, RouteSet] = {}
            for _, trial in expanded:
                unique.setdefault(self._key(trial), trial)
                if len(unique) >= self.config.beam_width * 3:
                    break
            states = list(unique.values())
            states.sort(key=lambda x: self._evaluate(x, full=False).score)
            states = states[:self.config.beam_width]

            new_beam: list[tuple[RouteSet, Evaluation]] = []
            top_full = states[:min(self.config.full_candidates_per_iteration, len(states))]
            for rank, trial in enumerate(top_full, 1):
                self._notify(f"  точная оценка {rank}/{len(top_full)}: сеть из {trial.route_count()} маршрутов")
                new_beam.append((trial, self._evaluate(trial, full=True)))
            for trial in states[len(top_full):]:
                new_beam.append((trial, self._evaluate(trial, full=False)))
            new_beam.sort(key=lambda x: x[1].score)
            beam = new_beam[:self.config.beam_width]
            best_network, _ = beam[0]
            best_eval = self._evaluate(best_network, full=True)
            history.append({"phase": "construct", "iteration": iteration + 1,
                            "routes": best_network.route_count(), "score": best_eval.score,
                            "beam_width": len(beam)})
            self._notify(f"  лучшая сеть: {best_network.route_count()} маршрутов, оценка {best_eval.score:.3f}")
            if best_network.route_count() >= self.config.max_routes:
                break

        network, current = min(((n, self._evaluate(n, full=True)) for n, _ in beam), key=lambda x: x[1].score)
        network, current = self._local_search(network, current, history)
        self._notify(f"Завершено: {network.route_count()} маршрутов, оценка {current.score:.3f}")
        return TNDPResult(network, current, history)

    def _local_search(self, network: RouteSet, current: Evaluation, history: list[dict]):
        if self._graph is None or network.route_count() == 0:
            return network, current
        for round_no in range(1, self.config.local_search_rounds + 1):
            self._notify(f"Локальный поиск: раунд {round_no}/{self.config.local_search_rounds}")
            trials = list(mutate_route_set(network, self._graph, self.config))
            ranked = sorted((self._evaluate(trial, full=False).score, trial, meta) for trial, meta in trials)
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
                            "new_route": list(best_meta["route"].nodes),
                            "frequency_vph": float(best_meta["route"].frequency_vph),
                            "vehicle_type": best_meta["route"].vehicle_type})
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
    vehicle_cost = 0.0
    for route in route_set.routes:
        seq = np.asarray(route.nodes, dtype=int)
        if len(seq) < 2:
            continue
        xy = np.asarray(node_xy_km)[seq]
        route_km += float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum())
        nodes = np.unique(seq)
        if len(nodes):
            served[np.ix_(nodes, nodes)] = True
        # A light vehicle-cost signal prevents the surrogate from systematically
        # preferring very large vehicles before exact assignment.
        from .vehicle_types import VEHICLE_TYPES
        vehicle_cost += VEHICLE_TYPES[route.vehicle_type].unit_cost_mln
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
             + overlap * config.duplication_weight
             + vehicle_cost * 0.001)
    return Evaluation(score=score, operator_cost=route_km, uncovered_demand=uncovered,
                      direct_demand_share=direct / total if total else 0.0,
                      metadata={"evaluator": "surrogate"})
