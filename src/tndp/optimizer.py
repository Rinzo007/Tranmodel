"""Whole-route TNDP optimizer with surrogate screening, local search and Pareto archive."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from .model import Evaluation, NetworkDesignConfig, Route, RouteSet
from .mutations import mutate_route_set
from .neighborhood import generate_network_moves
from .objective import apply_objective
from .pareto import compact_solution_record
from .pareto_archive import ParetoArchive

@dataclass
class TNDPResult:
    routes: RouteSet
    evaluation: Evaluation
    history: list[dict]
    pareto_archive: list[dict] | None = None

Evaluator = Callable[[RouteSet], Evaluation]
Progress = Callable[[str], None]

class TNDPOptimizer:
    """TNDP optimizer with constructive beam search, local search and bounded Pareto archive."""
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
        self.archive = ParetoArchive(max_size=100)

    @staticmethod
    def _key(network: RouteSet) -> tuple:
        return tuple(sorted((r.nodes, round(float(r.frequency_vph), 6), r.vehicle_type) for r in network.routes))

    def _record(self, network: RouteSet, ev: Evaluation) -> None:
        md = ev.metadata or {}
        coverage = float(md.get("coverage_share", md.get("coverage_800m", 0.0)) or 0.0)
        annual_cost = float(md.get("annual_total_cost_mln", md.get("annual_operating_cost_mln", 0.0)) or 0.0)
        peak_fleet = int(md.get("peak_fleet_reconciled", md.get("fleet", 0)) or 0)
        self.archive.add(compact_solution_record(
            score=ev.score, route_count=network.route_count(),
            annual_cost_mln=annual_cost, uncovered_demand=float(ev.uncovered_demand),
            coverage_share=coverage, user_cost=float(ev.user_cost),
            transfers=float(ev.transfers), fleet=peak_fleet,
            metadata={"key": repr(self._key(network)), "evaluator": md.get("evaluator", "unknown")}))

    def _evaluate(self, network: RouteSet, full: bool = True) -> Evaluation:
        cache = self._full_cache if full else self._fast_cache
        key = self._key(network)
        if key not in cache:
            raw = (self.evaluator if full else self.fast_evaluator)(network)
            cache[key] = apply_objective(network, raw, self.config)
            self._record(network, cache[key])
        return cache[key]

    def _notify(self, message: str) -> None:
        if self.progress: self.progress(message)
        print(f"[TNDP] {message}", flush=True)

    def _rank_additions(self, network: RouteSet, remaining: list[Route]):
        ranked = []
        for idx, route in enumerate(remaining, 1):
            if network.contains_nodes(route.nodes): continue
            trial = network.copy(); trial.add(route)
            ranked.append((self._evaluate(trial, full=False).score, trial, route))
            if idx % 25 == 0: self._notify(f"  быстрый отбор: {idx}/{len(remaining)}")
        ranked.sort(key=lambda x: x[0])
        return ranked

    def _construct_initial_beam(self):
        states = [RouteSet()]
        target = min(self.config.min_routes, self.config.max_routes, len(self.candidates))
        for _ in range(target):
            expanded = []
            for state in states:
                remaining = [r for r in self.candidates if not state.contains_nodes(r.nodes)]
                ranked = self._rank_additions(state, remaining)
                expanded.extend((s, t) for s, t, _ in ranked[:self.config.beam_expansion_per_state])
            if not expanded: break
            expanded.sort(key=lambda x: x[0])
            unique = {}
            for _, state in expanded:
                unique.setdefault(self._key(state), state)
                if len(unique) >= self.config.beam_width: break
            states = list(unique.values())
        return states

    def _screen_and_exact(self, trials, current):
        unique = {}
        for trial, meta in trials: unique.setdefault(self._key(trial), (trial, meta))
        ranked = sorted((self._evaluate(t, full=False).score, t, m) for t, m in unique.values())
        top_k = min(self.config.full_candidates_per_iteration, len(ranked))
        best_network, best_eval, best_meta = None, current, None
        for rank, (_, trial, meta) in enumerate(ranked[:top_k], 1):
            self._notify(f"  точная оценка соседа {rank}/{top_k}")
            ev = self._evaluate(trial, full=True)
            if ev.score + self.config.improvement_epsilon < best_eval.score:
                best_network, best_eval, best_meta = trial, ev, meta
        return best_network, best_eval, best_meta

    def solve(self, initial: RouteSet | None = None, graph=None) -> TNDPResult:
        self._graph = graph
        beam = [initial.copy()] if initial is not None and initial.route_count() else self._construct_initial_beam()
        if not beam: raise RuntimeError("TNDP could not construct an initial route network")
        scored = [(n, self._evaluate(n, True)) for n in beam]
        network, current = min(scored, key=lambda x: x[1].score)
        history = [{"phase": "start", "routes": network.route_count(), "score": current.score, "beam_width": len(beam), "pareto_size": len(self.archive.items)}]
        self._notify(f"Старт точной оценки: {network.route_count()} маршрутов")
        stagnant = 0
        for iteration in range(self.config.iterations):
            expanded = []
            for base, _ in scored[:self.config.beam_width]:
                remaining = [r for r in self.candidates if not base.contains_nodes(r.nodes)] if base.route_count() < self.config.max_routes else []
                if remaining:
                    ranked = self._rank_additions(base, remaining)
                    expanded.extend((s, t) for s, t, _ in ranked[:self.config.beam_expansion_per_state])
            expanded.sort(key=lambda x: x[0])
            states, seen = [], set()
            for _, state in expanded:
                k = self._key(state)
                if k in seen: continue
                seen.add(k); states.append(state)
                if len(states) >= self.config.beam_width: break
            if not states: break
            scored = [(n, self._evaluate(n, True)) for n in states]
            scored.sort(key=lambda x: x[1].score)
            improved = bool(scored and scored[0][1].score + self.config.improvement_epsilon < current.score)
            if improved: network, current, stagnant = scored[0][0], scored[0][1], 0
            else: stagnant += 1
            history.append({"phase": "construct", "iteration": iteration + 1, "routes": network.route_count(), "score": current.score, "beam_width": len(scored), "improved": improved, "feasible": current.metadata.get("feasible", True), "pareto_size": len(self.archive.items)})
            self._notify(f"  лучшая сеть: {network.route_count()} маршрутов, оценка {current.score:.3f}, Pareto={len(self.archive.items)}")
            if stagnant >= self.config.stagnation_rounds:
                self._notify(f"  остановка конструктивного поиска: {stagnant} итерации без улучшения"); break
        network, current = self._local_search(network, current, history)
        self._notify(f"Завершено: {network.route_count()} маршрутов, оценка {current.score:.3f}, Pareto={len(self.archive.items)}")
        return TNDPResult(network, current, history, self.archive.items.copy())

    def _local_search(self, network: RouteSet, current: Evaluation, history):
        if self._graph is None or not network.routes: return network, current
        stagnant = 0
        for round_no in range(1, self.config.local_search_rounds + 1):
            self._notify(f"Локальный поиск: раунд {round_no}/{self.config.local_search_rounds}")
            trials = list(mutate_route_set(network, self._graph, self.config))
            trials.extend(generate_network_moves(network, self.candidates, self.config))
            best_network, best_eval, best_meta = self._screen_and_exact(trials, current)
            if best_network is None:
                stagnant += 1; self._notify("  улучшений нет")
                if stagnant >= self.config.stagnation_rounds: break
                continue
            network, current, stagnant = best_network, best_eval, 0
            history.append({"phase": "local_search", "round": round_no, "routes": network.route_count(), "score": current.score,
                            "operation": best_meta.get("operation"), "index": best_meta.get("index"),
                            "new_route": list(best_meta["route"].nodes), "frequency_vph": float(best_meta["route"].frequency_vph),
                            "vehicle_type": best_meta["route"].vehicle_type, "feasible": current.metadata.get("feasible", True), "pareto_size": len(self.archive.items)})
            self._notify(f"  улучшение: {current.score:.3f}")
        return network, current
