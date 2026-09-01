from src.tndp.vehicle_types import VEHICLE_TYPES, calculate_route_operations
from src.tndp.economics_adapter import calculate_annual_route_economics
from src.tndp.interval_profile import DEFAULT_INTERVAL_PROFILE, as_frequency_profile


def test_authoritative_economics_has_consistent_total():
    result = calculate_annual_route_economics(route_length_km=10.0,max_section_flow_pph=400.0,vehicle_type="liaz")
    costs = result["economics"]
    parts = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln")
    expected = sum(float(costs.get(k, 0.0)) for k in parts)
    assert abs(float(costs["total_annual_mln"]) - expected) < 1e-6 * max(1.0, expected)
    assert result["fleet"] >= 1
    assert result["annual_mileage_km"] > 0
    assert result["annual_in_service_hours"] > 0


def test_vehicle_wrapper_matches_canonical_physics():
    profile = as_frequency_profile(DEFAULT_INTERVAL_PROFILE)
    for code in ("liaz", "kamaz_charge_terminal", "tm_vityaz"):
        wrapper = calculate_route_operations(route_length_km=12.0,max_section_flow_pph=300.0,vehicle_type=code,frequency_profile=profile)
        canonical = calculate_annual_route_economics(route_length_km=12.0,max_section_flow_pph=300.0,vehicle_type=code,frequency_profile=profile)
        assert wrapper["interval_min"] == canonical["interval_min"]
        assert wrapper["fleet"] == canonical["fleet"]
        assert abs(wrapper["annual_mileage_km"] - canonical["annual_mileage_km"]) < 1e-9
        assert VEHICLE_TYPES[code].capacity > 0


def test_profile_reference_hours_are_145():
    total = sum(hours * factor for hours, factor in as_frequency_profile(DEFAULT_INTERVAL_PROFILE))
    assert abs(total - 14.5) < 1e-9
