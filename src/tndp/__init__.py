"""Transit Network Design Problem (TNDP) solver for Tranmodel.

The solver generates candidate public-transport routes from an OD matrix,
optimizes a route set, and can delegate full network evaluation to AequilibraE.
"""

from .model import NetworkDesignConfig, Route, RouteSet
from .corridors import DemandCorridor, extract_demand_corridors
from .candidates import generate_route_candidates
from .optimizer import TNDPOptimizer, TNDPResult
from .aequilibrae_eval import AequilibraEEvaluationError, evaluate_route_set_aequilibrae
from .pareto import ObjectiveVector, dominates, pareto_front, vector_from_evaluation
from .service_indicators import ServiceIndicators, safe_share, capacity_excess

__all__ = [
    "NetworkDesignConfig",
    "Route",
    "RouteSet",
    "DemandCorridor",
    "extract_demand_corridors",
    "generate_route_candidates",
    "TNDPOptimizer",
    "TNDPResult",
    "AequilibraEEvaluationError",
    "evaluate_route_set_aequilibrae",
    "ObjectiveVector",
    "dominates",
    "pareto_front",
    "vector_from_evaluation",
    "ServiceIndicators",
    "safe_share",
    "capacity_excess",
]
