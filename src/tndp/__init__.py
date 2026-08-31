"""Transit Network Design Problem (TNDP) solver for Tranmodel."""

from .model import NetworkDesignConfig, Route, RouteSet
from .corridors import DemandCorridor, extract_demand_corridors
from .candidates import generate_route_candidates
from .optimizer import TNDPOptimizer, TNDPResult
from .aequilibrae_eval import AequilibraEEvaluationError, evaluate_route_set_aequilibrae
from .pareto import ObjectiveVector, dominates, pareto_front, vector_from_evaluation
from .pareto_archive import ParetoArchive
from .service_indicators import ServiceIndicators, safe_share, capacity_excess
from .service_metrics import ServiceMetrics, generalized_cost

__all__ = [
    "NetworkDesignConfig", "Route", "RouteSet", "DemandCorridor",
    "extract_demand_corridors", "generate_route_candidates", "TNDPOptimizer",
    "TNDPResult", "AequilibraEEvaluationError", "evaluate_route_set_aequilibrae",
    "ObjectiveVector", "dominates", "pareto_front", "vector_from_evaluation",
    "ParetoArchive", "ServiceIndicators", "safe_share", "capacity_excess",
    "ServiceMetrics", "generalized_cost",
]
