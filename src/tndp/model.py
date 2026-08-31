"""Core data structures for transit network design."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Route:
    """A directed transit route represented by ordered network node IDs."""

    nodes: tuple[int, ...]
    route_id: str | None = None
    frequency_vph: float = 6.0

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("A route must contain at least two nodes")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("A route cannot repeat a node")
        if self.frequency_vph <= 0:
            raise ValueError("frequency_vph must be positive")

    def reversed(self, route_id: str | None = None) -> "Route":
        return Route(tuple(reversed(self.nodes)), route_id or self.route_id, self.frequency_vph)


@dataclass
class RouteSet:
    """A collection of routes with helpers used by the optimizer."""

    routes: list[Route] = field(default_factory=list)

    def add(self, route: Route) -> None:
        if route not in self.routes:
            self.routes.append(route)

    def copy(self) -> "RouteSet":
        return RouteSet(list(self.routes))

    def contains_nodes(self, nodes: Iterable[int]) -> bool:
        target = tuple(nodes)
        return any(r.nodes == target for r in self.routes)

    def route_count(self) -> int:
        return len(self.routes)

    def unique_undirected_signatures(self) -> set[tuple[int, ...]]:
        return {min(r.nodes, tuple(reversed(r.nodes))) for r in self.routes}


@dataclass(frozen=True, slots=True)
class NetworkDesignConfig:
    """Constraints and weights for TNDP search."""

    min_routes: int = 10
    max_routes: int = 80
    min_stops: int = 5
    max_stops: int = 40
    min_route_demand: float = 50.0
    max_route_length_km: float = 25.0
    min_route_length_km: float = 2.0
    max_detour_ratio: float = 1.45
    candidate_limit_per_corridor: int = 8
    corridor_top_pairs: int = 300
    corridor_distance_km: float = 8.0
    iterations: int = 100
    improvement_epsilon: float = 1e-6
    transfer_penalty_min: float = 8.0
    wait_weight: float = 1.5
    in_vehicle_weight: float = 1.0
    walk_weight: float = 2.0
    transfer_weight: float = 1.0
    operator_route_km_weight: float = 0.02
    uncovered_demand_weight: float = 3.0
    duplication_weight: float = 1.0
