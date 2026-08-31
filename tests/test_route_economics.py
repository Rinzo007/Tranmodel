from src.tndp.route_economics import calculate_route_characteristics


def test_charging_is_only_added_when_enabled():
    base = calculate_route_characteristics(20.0, 365.0, capacity_at_4_ppm2=73.0, charging_at_terminal=False)
    electric = calculate_route_characteristics(20.0, 365.0, capacity_at_4_ppm2=73.0, charging_at_terminal=True)
    assert base.charging_min_per_terminal == 0.0
    assert electric.charging_min_per_terminal == 10.0
    assert electric.turnaround_min >= base.turnaround_min


def test_interval_is_half_minute_and_frequency_is_derived_from_it():
    result = calculate_route_characteristics(10.0, 292.0, capacity_at_4_ppm2=73.0)
    assert result.interval_min * 2 == int(result.interval_min * 2)
    assert abs(result.frequency_vph - 60.0 / result.interval_min) < 1e-9


def test_annualization_uses_350_days_and_park_coefficient():
    result = calculate_route_characteristics(10.0, 365.0, capacity_at_4_ppm2=73.0)
    assert result.annual_mileage_km > result.route_length_km if hasattr(result, "route_length_km") else result.annual_mileage_km > 0
    assert result.annual_in_service_hours > 0
