"""Whole-route TNDP optimizer with surrogate screening and bounded exact search."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from .model import Evaluation, NetworkDesignConfig, Route, RouteSet
from .mutations import mutate_route_set
from .neighborhood import generate_network_moves
from .objective import apply_objective

@dataclass
class TNDPResult:
    routes: RouteSet
    evaluation: Evaluation
    history: list[dict]

Evaluator = Callable[[RouteSet], Evaluation]
Progress = Callable[[str], None]

class TNDPOptimizer:
    """TNDP optimizer with constructive beam search and variable-size local search."""
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
            raw = (self.evaluator if full else self.fast_evaluator)(network)
            cache[key] = apply_objective(network, raw, self.config)
        return cache[key]

    def _notify(self, message: str) -> None:
        if self.progress: self.progress(message)
        print(f"[TNDP] {message}", flush=True)

    def _rank_additions(self, network: RouteSet, remaining: list[Route]) -> list[tuple[float, RouteSet, Route]]:
        ranked = []
        for idx, route in enumerate(remaining, 1):
            if network.contains_nodes(route.nodes): continue
            trial = network.copy(); trial.add(route)
            ranked.append((self._evaluate(trial, full=False).score, trial, route))
            if idx % 25 == 0: self._notify(f"  быстрый отбор: {idx}/{len(remaining)}")
        ranked.sort(key=lambda x: x[0])
        return ranked

    def _construct_initial_beam(self) -> list[RouteSet]:
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

    def _screen_and_exact(self, trials: list[tuple[RouteSet, dict]], current: Evaluation):
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
        history = [{"phase": "start", "routes": network.route_count(), "score": current.score, "beam_width": len(beam)}]
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
            history.append({"phase": "construct", "iteration": iteration + 1, "routes": network.route_count(), "score": current.score, "beam_width": len(scored), "improved": improved, "feasible": current.metadata.get("feasible", True)})
            self._notify(f"  лучшая сеть: {network.route_count()} маршрутов, оценка {current.score:.3f}")
            if stagnant >= self.config.stagnation_rounds:
                self._notify(f"  остановка конструктивного поиска: {stagnant} итерации без улучшения"); break
        network, current = self._local_search(network, current, history)
        self._notify(f"Завершено: {network.route_count()} маршрутов, оценка {current.score:.3f}")
        return TNDPResult(network, current, history)

    def _local_search(self, network: RouteSet, current: Evaluation, history: list[dict]):
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
                            "vehicle_type": best_meta["route"].vehicle_type, "feasible": current.metadata.get("feasible", True)})
            self._notify(f"  улучшение: {current.score:.3f}")
        return network, current


def surrogate_evaluator(demand, node_xy_km, route_set: RouteSet, config: NetworkDesignConfig | None = None, *args, **kwargs) -> Evaluation:
    import numpy as np
    from .vehicle_types import calculate_route_operations
    config = config or NetworkDesignConfig(); matrix = np.asarray(demand, dtype=float); total = float(matrix.sum())
    if not route_set.routes: return Evaluation(score=total * config.uncovered_demand_weight if total else 0.0, uncovered_demand=total, metadata={"evaluator": "surrogate", "empty_network": True})
    route_km = 0.0; served = np.zeros(matrix.shape, dtype=bool); vehicle_cost = 0.0; annual_cost = 0.0; annual_amortization = 0.0; fleet = 0
    for route in route_set.routes:
        seq = np.asarray(route.nodes, dtype=int)
        if len(seq) < 2: continue
        xy = np.asarray(node_xy_km)[seq]; length = float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum()); route_km += length
        nodes = np.unique(seq)
        if len(nodes): served[np.ix_(nodes, nodes)] = True
        op = calculate_route_operations(route_length_km=length, max_section_flow_pph=max(route.max_section_flow_pph, 0.0), vehicle_type=route.vehicle_type,
            speed_kmh=config.speed_kmh, interval_reserve_sec=config.interval_reserve_sec, terminal_delay_reserve=config.terminal_delay_reserve,
            charging_min_per_terminal=config.charging_min_per_terminal, annual_days=config.annual_days, park_trip_coefficient=config.park_trip_coefficient, frequency_profile=config.frequency_profile)
        annual_cost += float(op["annual_fleet_contract_cost_mln"]); annual_amortization += float(op["annual_fleet_amortization_mln"]); fleet += int(op["fleet"]); vehicle_cost += float(op["one_off_fleet_cost_mln"])
    np.fill_diagonal(served, False); direct = float(matrix[served].sum()) if total else 0.0; uncovered = max(0.0, total - direct)
    node_counts = {}
    for route in route_set.routes:
        for node in set(route.nodes): node_counts[node] = node_counts.get(node, 0) + 1
    overlap = sum(max(0, c - 1) for c in node_counts.values())
    score = uncovered * config.uncovered_demand_weight + route_km * config.operator_route_km_weight + overlap * config.duplication_weight + vehicle_cost * 0.001
    return Evaluation(score=score, operator_cost=route_km, uncovered_demand=uncovered, direct_demand_share=direct / total if total else 0.0,
        metadata={"evaluator": "surrogate", "fleet": fleet, "annual_contract_cost_mln": annual_cost, "annual_amortization_mln": annual_amortization})
