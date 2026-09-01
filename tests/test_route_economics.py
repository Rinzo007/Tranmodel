from math import isclose

from src.tndp.route_economics import calculate_route_characteristics
from src.tndp.vehicle_types import VEHICLE_TYPES


def test_charging_is_only_added_when_enabled():
    base = calculate_route_characteristics(
        20.0, 365.0, capacity_at_4_ppm2=73.0, charging_at_terminal=False
    )
    electric = calculate_route_characteristics(
        20.0, 365.0, capacity_at_4_ppm2=73.0, charging_at_terminal=True
    )
    assert base.charging_min_per_terminal == 0.0
    assert electric.charging_min_per_terminal == 10.0
    assert electric.turnaround_min >= base.turnaround_min


def test_interval_has_half_minute_precision_and_frequency_is_derived_from_it():
    result = calculate_route_characteristics(
        10.0, 292.0, capacity_at_4_ppm2=73.0
    )
    assert result.interval_min * 2 == int(result.interval_min * 2)
    assert isclose(result.frequency_vph, 60.0 / result.interval_min)


def test_operating_formulas_match_methodology():
    result = calculate_route_characteristics(
        10.0, 365.0, capacity_at_4_ppm2=73.0,
        annual_days=350, park_trip_coefficient=0.90,
    )
    expected_km = result.route_length_km * result.daily_trips / 0.90 * 350
    expected_hours = result.turnaround_min * result.daily_trips / 0.90 * 350 / 60.0
    assert isclose(result.annual_mileage_km, expected_km)
    assert isclose(result.annual_in_service_hours, expected_hours)


def test_default_profile_has_14_5_effective_peak_hours():
    result = calculate_route_characteristics(
        10.0, 365.0, capacity_at_4_ppm2=73.0
    )
    # 1*0.8 + 2*1 + 7.5*0.8 + 3*1 + 1.5*0.8 + 3*0.5 = 14.5
    expected_daily_trips = 14.5 * result.frequency_vph
    assert isclose(result.daily_trips, expected_daily_trips)


def test_all_catalogue_vehicle_types_have_valid_economic_parameters():
    assert len(VEHICLE_TYPES) == 16
    for vehicle in VEHICLE_TYPES.values():
        assert vehicle.capacity > 0
        assert vehicle.unit_cost_mln > 0
        assert 0 < vehicle.technical_readiness <= 1
        assert vehicle.driver_hour_with_charges > 0
        result = calculate_route_characteristics(
            10.0,
            vehicle.capacity,
            capacity_at_4_ppm2=vehicle.capacity,
            technical_readiness=vehicle.technical_readiness,
            charging_at_terminal=vehicle.charging_at_terminal,
        )
        assert result.fleet >= result.release
        assert result.annual_mileage_km > 0
        assert result.annual_in_service_hours > 0
