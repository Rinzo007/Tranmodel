import networkx as nx

from tndp.model import Evaluation, NetworkDesignConfig, Route, RouteSet
from tndp.optimizer import TNDPOptimizer
from tndp.route_loads import select_vehicle_for_route


def _graph():
    g = nx.Graph()
    for a, b, km in [(1, 2, 2.0), (2, 3, 2.0), (3, 4, 2.0), (4, 5, 2.0)]:
        g.add_edge(a, b, length_km=km, time=km / 18.0 * 60.0)
    return g


def test_service_selection_rejects_under_capacity_vehicle():
    code, details = select_vehicle_for_route(
        max_section_flow_pph=100.0,
        route_length_km=16.0,
        allowed_vehicle_types=("liaz", "liaz_obk", "tm_lvenok", "tm_vityaz"),
    )
    assert details["capacity_ok"]
    assert details["capacity"] >= 100.0
    assert code == "tm_vityaz"


def test_optimizer_reconciles_vehicle_and_frequency_after_route_change():
    config = NetworkDesignConfig(min_routes=1, max_routes=1, allowed_vehicle_types=("liaz", "liaz_obk", "tm_vityaz"))
    optimizer = TNDPOptimizer(
        candidates=[],
        evaluator=lambda rs: Evaluation(score=0.0),
        config=config,
    )
    optimizer._graph = _graph()
    stale = Route((1, 2, 3, 4, 5), frequency_vph=3.0, max_section_flow_pph=100.0, vehicle_type="liaz")
    normalized = optimizer._reconcile_route(stale)
    assert normalized.vehicle_type == "tm_vityaz"
    assert normalized.frequency_vph > 0
    assert normalized.frequency_vph != stale.frequency_vph


def test_optimizer_evaluation_uses_reconciled_route_set():
    seen = []
    config = NetworkDesignConfig(min_routes=1, max_routes=1, allowed_vehicle_types=("liaz", "liaz_obk", "tm_vityaz"))
    def evaluator(rs):
        seen.append(rs.routes[0])
        return Evaluation(score=1.0)
    optimizer = TNDPOptimizer(candidates=[], evaluator=evaluator, config=config)
    optimizer._graph = _graph()
    stale = RouteSet([Route((1, 2, 3, 4, 5), frequency_vph=3.0, max_section_flow_pph=100.0, vehicle_type="liaz")])
    optimizer._evaluate(stale, full=True)
    assert seen
    assert seen[0].vehicle_type == "tm_vityaz"
    assert seen[0].frequency_vph > 0
