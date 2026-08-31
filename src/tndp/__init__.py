"""Transit Network Design Problem (TNDP) solver for Tranmodel."""

from .model import NetworkDesignConfig, Route, RouteSet, Evaluation
from .corridors import DemandCorridor, extract_demand_corridors
from .candidates import generate_route_candidates
from .optimizer import TNDPOptimizer, TNDPResult
from .aequilibrae_eval import AequilibraEEvaluationError, evaluate_route_set_aequilibrae
from .pareto import ObjectiveVector, dominates, pareto_front, vector_from_evaluation
from .pareto_archive import ParetoArchive
from .service_indicators import ServiceIndicators, safe_share, capacity_excess
from .service_metrics import ServiceMetrics, generalized_cost
from .multi_period import PeriodPlan, build_period_plan, summarize_period_plan
from .operating_plan import build_network_operating_plan

__all__ = [
    "NetworkDesignConfig", "Route", "RouteSet", "Evaluation", "DemandCorridor",
    "extract_demand_corridors", "generate_route_candidates", "TNDPOptimizer",
    "TNDPResult", "AequilibraEEvaluationError", "evaluate_route_set_aequilibrae",
    "ObjectiveVector", "dominates", "pareto_front", "vector_from_evaluation",
    "ParetoArchive", "ServiceIndicators", "safe_share", "capacity_excess",
    "ServiceMetrics", "generalized_cost", "PeriodPlan", "build_period_plan",
    "summarize_period_plan", "build_network_operating_plan",
]
