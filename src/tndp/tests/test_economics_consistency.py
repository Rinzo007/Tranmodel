from src.tndp.economics_adapter import calculate_annual_route_economics


def test_authoritative_economics_has_consistent_total():
    result = calculate_annual_route_economics(
        route_length_km=10.0,
        max_section_flow_pph=400.0,
        vehicle_type="liaz",
    )
    costs = result["economics"]
    parts = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln")
    expected = sum(float(costs.get(k, 0.0)) for k in parts)
    assert abs(float(costs["total_annual_mln"]) - expected) < 1e-6 * max(1.0, expected)
    assert result["fleet"] >= 1
    assert result["annual_mileage_km"] > 0
    assert result["annual_in_service_hours"] > 0
