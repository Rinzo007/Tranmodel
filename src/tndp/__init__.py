"""Transit Network Design Problem (TNDP) solver for Tranmodel."""
from .model import NetworkDesignConfig, Route, RouteSet, Evaluation
from .corridors import DemandCorridor, extract_demand_corridors
from .candidates import generate_route_candidates
from .optimizer import TNDPOptimizer, TNDPResult
from .aequilibrae_eval import AequilibraEEvaluationError, evaluate_route_set_aequilibrae
from .pareto import dominates, pareto_front
from .pareto_archive import ParetoArchive
from .service_metrics import ServiceMetrics, generalized_cost
from .interval_profile import IntervalPeriod, DEFAULT_INTERVAL_PROFILE, daily_frequency_factor, as_frequency_profile, validate_profile
from .multi_period import PeriodPlan, build_period_plan, summarize_period_plan
from .operating_plan import build_network_operating_plan
from .period_assignment import PeriodDemand, PeriodAssignment, build_period_demands, aggregate_period_assignments
from .period_costs import route_period_costs
from .cost_aggregation import aggregate_route_costs, aggregate_network_costs, aggregate_peak_fleet
from .multiperiod_assignment import evaluate_route_set_aequilibrae_periods
from .multiperiod_cache import evaluate_cached_period, period_cache_key, cache_stats
from .multiperiod_cached_eval import evaluate_route_set_aequilibrae_periods_cached
from .peak_fleet import PeriodRouteOperation, reconcile_route_periods
from .period_vehicle_plan import build_route_vehicle_plan, build_route_vehicle_plan_auto, build_network_vehicle_plan
from .period_network_plan import reconcile_period_network
from .economics_source import calculate_annual_route_economics, REQUIRED_COST_KEYS
from .economics_validation import validate_route_economics
from .economic_catalog_validation import validate_economic_catalogue, assert_economic_catalogue

__all__ = [
    "NetworkDesignConfig", "Route", "RouteSet", "Evaluation", "DemandCorridor", "extract_demand_corridors", "generate_route_candidates",
    "TNDPOptimizer", "TNDPResult", "AequilibraEEvaluationError", "evaluate_route_set_aequilibrae", "dominates",
    "pareto_front", "ParetoArchive", "ServiceMetrics",
    "generalized_cost", "IntervalPeriod", "DEFAULT_INTERVAL_PROFILE", "daily_frequency_factor", "as_frequency_profile", "validate_profile",
    "PeriodPlan", "build_period_plan", "summarize_period_plan", "build_network_operating_plan", "PeriodDemand", "PeriodAssignment",
    "build_period_demands", "aggregate_period_assignments", "route_period_costs", "aggregate_route_costs", "aggregate_network_costs",
    "aggregate_peak_fleet", "evaluate_route_set_aequilibrae_periods", "evaluate_cached_period", "period_cache_key", "cache_stats",
    "evaluate_route_set_aequilibrae_periods_cached", "PeriodRouteOperation", "reconcile_route_periods",
    "build_route_vehicle_plan", "build_route_vehicle_plan_auto", "build_network_vehicle_plan", "reconcile_period_network",
    "calculate_annual_route_economics", "REQUIRED_COST_KEYS", "validate_route_economics", "validate_economic_catalogue", "assert_economic_catalogue",
]
