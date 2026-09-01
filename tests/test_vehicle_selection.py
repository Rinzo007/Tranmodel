from tndp.vehicle_selection import evaluate_vehicle_alternatives, select_vehicle_type
from tndp.vehicle_types import VEHICLE_TYPES


def test_all_vehicle_types_are_evaluated():
    result = evaluate_vehicle_alternatives(route_length_km=10.0, max_section_flow_pph=100.0)
    assert len(result) == len(VEHICLE_TYPES) == 16
    assert all(x["release"] >= 1 for x in result)
    assert all(x["fleet"] >= 1 for x in result)


def test_capacity_objective_selects_smallest_feasible_vehicle():
    result = select_vehicle_type(
        route_length_km=10.0,
        max_section_flow_pph=90.0,
        objective="capacity",
    )
    assert result["selected"]["capacity_ok"]
    assert result["selected"]["capacity"] == 93


def test_cost_objective_selects_only_capacity_feasible_vehicle():
    result = select_vehicle_type(
        route_length_km=10.0,
        max_section_flow_pph=160.0,
        objective="cost",
    )
    assert result["selected"]["capacity_ok"]
    assert result["selected"]["capacity"] >= 160


def test_over_capacity_flow_returns_explicit_warning():
    result = select_vehicle_type(route_length_km=10.0, max_section_flow_pph=1000.0)
    assert not result["selected"]["capacity_ok"]
    assert "capacity_warning" in result["selected"]
