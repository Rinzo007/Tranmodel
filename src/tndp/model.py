"""Core data structures and constraints for Transit Network Design."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Route:
    """Directed transit route represented by ordered stop indices."""

    nodes: tuple[int, ...]
    route_id: str | None = None
    frequency_vph: float = 6.0

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("A route must contain at least two nodes")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("A route cannot repeat a stop")
        if self.frequency_vph <= 0:
            raise ValueError("frequency_vph must be positive")

    def reversed(self, route_id: str | None = None) -> "Route":
        return Route(tuple(reversed(self.nodes)), route_id or self.route_id, self.frequency_vph)

    def with_nodes(self, nodes: Iterable[int]) -> "Route":
        return Route(tuple(int(n) for n in nodes), self.route_id, self.frequency_vph)

    def with_frequency(self, frequency_vph: float) -> "Route":
        return Route(self.nodes, self.route_id, float(frequency_vph))


@dataclass
class RouteSet:
    """A collection of directed routes with exact duplicate handling."""

    routes: list[Route] = field(default_factory=list)

    def add(self, route: Route) -> None:
        if route not in self.routes and not self.contains_nodes(route.nodes):
            self.routes.append(route)

    def remove_at(self, index: int) -> None:
        del self.routes[index]

    def copy(self) -> "RouteSet":
        return RouteSet(list(self.routes))

    def route_count(self) -> int:
        return len(self.routes)

    def contains_nodes(self, nodes: Iterable[int]) -> bool:
        target = tuple(int(n) for n in nodes)
        return any(r.nodes == target for r in self.routes)

    def unique_undirected_signatures(self) -> set[tuple[int, ...]]:
        return {min(r.nodes, tuple(reversed(r.nodes))) for r in self.routes}


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Objective-function result for a complete route-set evaluation."""

    score: float
    user_cost: float = 0.0
    operator_cost: float = 0.0
    uncovered_demand: float = 0.0
    transfers: float = 0.0
    direct_demand_share: float = 0.0
    capacity_excess: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NetworkDesignConfig:
    """Constraints and weights for TNDP search."""

    min_routes: int = 10
    max_routes: int = 40
    min_stops: int = 5
    max_stops: int = 40
    min_route_length_km: float = 2.0
    max_route_length_km: float = 25.0
    max_detour_ratio: float = 1.45
    candidate_limit_per_corridor: int = 8
    corridor_top_pairs: int = 300
    corridor_distance_km: float = 8.0
    iterations: int = 100

    min_frequency_vph: float = 3.0
    max_frequency_vph: float = 15.0
    vehicle_capacity: float = 73.0
    target_load_factor: float = 0.85
    peak_hours: float = 4.0

    transfer_penalty_min: float = 8.0
    wait_weight: float = 1.5
    in_vehicle_weight: float = 1.0
    walk_weight: float = 2.0
    transfer_weight: float = 1.0
    operator_route_km_weight: float = 0.02
    uncovered_demand_weight: float = 10.0
    capacity_excess_weight: float = 5.0
    duplication_weight: float = 2.0

    improvement_epsilon: float = 1e-6
    local_search_rounds: int = 4
    mutations_per_route: int = 12
    full_evaluation: bool = True
    # Number of fast-scored additions/mutations that receive the expensive
    # AequilibraE evaluation. Keeping this >1 avoids committing to a route
    # solely because the surrogate happened to rank it first.
    full_candidates_per_iteration: int = 3

    def validate(self) -> None:
        if self.min_routes < 0 or self.max_routes < self.min_routes:
            raise ValueError("Invalid route-count bounds")
        if self.min_stops < 2 or self.max_stops < self.min_stops:
            raise ValueError("Invalid stop-count bounds")
        if self.min_route_length_km <= 0 or self.max_route_length_km < self.min_route_length_km:
            raise ValueError("Invalid route-length bounds")
        if not (0 < self.min_frequency_vph <= self.max_frequency_vph):
            raise ValueError("Invalid frequency bounds")
        if not (0 < self.target_load_factor <= 1):
            raise ValueError("target_load_factor must be in (0, 1]")
        if self.vehicle_capacity <= 0:
            raise ValueError("vehicle_capacity must be positive")
        if self.full_candidates_per_iteration < 1:
            raise ValueError("full_candidates_per_iteration must be positive")
