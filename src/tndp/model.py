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
    max_section_flow_pph: float = 0.0
    vehicle_type: str = "bus"

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("A route must contain at least two nodes")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("A route cannot repeat a stop")
        if self.frequency_vph <= 0:
            raise ValueError("frequency_vph must be positive")
        if self.max_section_flow_pph < 0:
            raise ValueError("max_section_flow_pph cannot be negative")
        if self.vehicle_type not in {"bus", "electric_transit"}:
            raise ValueError("vehicle_type must be 'bus' or 'electric_transit'")

    def reversed(self, route_id: str | None = None) -> "Route":
        return Route(tuple(reversed(self.nodes)), route_id or self.route_id, self.frequency_vph, self.max_section_flow_pph, self.vehicle_type)

    def with_nodes(self, nodes: Iterable[int]) -> "Route":
        return Route(tuple(int(n) for n in nodes), self.route_id, self.frequency_vph, self.max_section_flow_pph, self.vehicle_type)

    def with_frequency(self, frequency_vph: float) -> "Route":
        return Route(self.nodes, self.route_id, float(frequency_vph), self.max_section_flow_pph, self.vehicle_type)

    def with_flow(self, max_section_flow_pph: float) -> "Route":
        return Route(self.nodes, self.route_id, self.frequency_vph, float(max_section_flow_pph), self.vehicle_type)

    def with_vehicle_type(self, vehicle_type: str) -> "Route":
        return Route(self.nodes, self.route_id, self.frequency_vph, self.max_section_flow_pph, vehicle_type)


@dataclass
class RouteSet:
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

    speed_kmh: float = 18.0
    capacity_at_4_ppm2: float = 73.0
    interval_reserve_sec: float = 20.0
    terminal_delay_reserve: float = 0.08
    charging_min_per_terminal: float = 10.0
    bus_technical_readiness: float = 0.80
    electric_technical_readiness: float = 0.90
    park_trip_coefficient: float = 0.90
    annual_days: int = 350
    frequency_profile: tuple[tuple[float, float], ...] = (
        (3.0, 1.00), (6.0, 0.75), (4.0, 1.00), (3.0, 0.60), (8.0, 0.30)
    )
    default_vehicle_type: str = "bus"

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
    full_candidates_per_iteration: int = 3
    beam_width: int = 3
    beam_expansion_per_state: int = 4
    full_states_per_iteration: int = 3

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
        if self.vehicle_capacity <= 0 or self.capacity_at_4_ppm2 <= 0:
            raise ValueError("vehicle capacity must be positive")
        if self.speed_kmh <= 0 or self.interval_reserve_sec < 0:
            raise ValueError("Invalid speed/interval reserve")
        if self.terminal_delay_reserve < 0 or self.charging_min_per_terminal < 0:
            raise ValueError("Invalid terminal assumptions")
        if not (0 < self.bus_technical_readiness <= 1 and 0 < self.electric_technical_readiness <= 1):
            raise ValueError("Technical readiness must be in (0, 1]")
        if not (0 < self.park_trip_coefficient <= 1) or self.annual_days <= 0:
            raise ValueError("Invalid annualization parameters")
        if not self.frequency_profile or any(h < 0 or m < 0 for h, m in self.frequency_profile):
            raise ValueError("Invalid frequency profile")
        if self.default_vehicle_type not in {"bus", "electric_transit"}:
            raise ValueError("default_vehicle_type must be 'bus' or 'electric_transit'")
        if self.full_candidates_per_iteration < 1:
            raise ValueError("full_candidates_per_iteration must be positive")
        if self.beam_width < 1 or self.beam_expansion_per_state < 1:
            raise ValueError("beam_width and beam_expansion_per_state must be positive")
        if self.full_states_per_iteration < 1:
            raise ValueError("full_states_per_iteration must be positive")
