"""Transit Network Design Problem (TNDP) solver for Tranmodel.

The solver generates candidate public-transport routes from an OD matrix,
optimizes a route set, and can delegate network assignment to AequilibraE.
"""

from .model import NetworkDesignConfig, Route, RouteSet
from .corridors import DemandCorridor, extract_demand_corridors
from .candidates import generate_route_candidates
from .optimizer import TNDPOptimizer, TNDPResult

__all__ = [
    "NetworkDesignConfig",
    "Route",
    "RouteSet",
    "DemandCorridor",
    "extract_demand_corridors",
    "generate_route_candidates",
    "TNDPOptimizer",
    "TNDPResult",
]
