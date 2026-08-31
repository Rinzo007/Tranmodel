from src.tndp.economics_core import calculate_annual_route_economics as canonical
from src.tndp.economics_source import calculate_annual_route_economics as source
from src.tndp.economics_adapter import calculate_annual_route_economics as adapter
from src.tndp.route_economics_single import calculate_annual_route_economics as legacy


def test_all_economics_entry_points_are_same_function():
    assert source is canonical
    assert adapter is canonical
    assert legacy is canonical


def test_canonical_economics_has_expected_components():
    result = canonical(route_length_km=10.0, max_section_flow_pph=400.0, vehicle_type="liaz")
    costs = result["economics"]
    assert result["frequency_vph"] > 0
    assert result["fleet"] >= 1
    assert costs["total_annual_mln"] >= costs["fuel_energy_mln"] + costs["repair_mln"] + costs["crew_mln"]
    assert result["annual_total_cost_mln"] == costs["total_annual_mln"]
