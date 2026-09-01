"""Stage 5: regression matrix for all rolling-stock types and six service periods."""
from __future__ import annotations

import math

import pytest

from tndp.economics_core import calculate_annual_route_economics
from tndp.route_economics import DEFAULT_FREQUENCY_PROFILE
from tndp.vehicle_types import VEHICLE_TYPES

EXPECTED_VEHICLES = {
    "ford_transit": (18, 0.80),
    "gazelle_city": (18, 0.80),
    "paz": (43, 0.80),
    "liaz": (68, 0.80),
    "liaz_gas": (68, 0.80),
    "liaz_obk": (93, 0.80),
    "liaz_obk_gas": (93, 0.80),
    "kamaz_charge_terminal": (72, 0.80),
    "admiral_bk": (73, 0.90),
    "admiral_obk": (98, 0.90),
    "tuah_bk": (73, 0.90),
    "tuah_obk": (98, 0.90),
    "tm_lvenok": (95, 0.90),
    "tm_vityaz": (162, 0.90),
    "tm_2x_bk": (190, 0.90),
    "tm_3x_bk": (285, 0.90),
}

# The source methodology defines six frequency periods whose durations sum to 18 h.
# (The supplied table's "14.5" is the weighted number of hours at one peak-rate
# trip/hour; the actual service span is 18 h.)
EXPECTED_PROFILE = ((1.0, 0.8), (2.0, 1.0), (7.5, 0.8), (3.0, 1.0), (1.5, 0.8), (3.0, 0.5))


@pytest.mark.parametrize("vehicle_code", list(EXPECTED_VEHICLES))
def test_all_16_vehicle_types_have_expected_capacity_and_ktg(vehicle_code: str):
    vehicle = VEHICLE_TYPES[vehicle_code]
    expected_capacity, expected_ktg = EXPECTED_VEHICLES[vehicle_code]
    assert vehicle.capacity == expected_capacity
    assert vehicle.technical_readiness == expected_ktg
    assert vehicle.unit_cost_mln > 0
    assert vehicle.service_life_years > 0
    assert vehicle.annual_contract_cost_mln > 0
    assert vehicle.annual_amortization_mln > 0


def test_six_period_profile_is_canonical():
    assert DEFAULT_FREQUENCY_PROFILE == EXPECTED_PROFILE
    assert sum(hours for hours, _ in DEFAULT_FREQUENCY_PROFILE) == pytest.approx(18.0)
    assert sum(hours * factor for hours, factor in DEFAULT_FREQUENCY_PROFILE) == pytest.approx(14.5)


@pytest.mark.parametrize("vehicle_code", list(EXPECTED_VEHICLES))
def test_each_vehicle_passes_full_six_period_economic_calculation(vehicle_code: str):
    result = calculate_annual_route_economics(
        vehicle_type=vehicle_code,
        route_length_km=20.0,
        max_section_flow_pph=500.0,
    )
    assert result["interval_min"] >= 0.5
    assert result["release"] >= 1
    assert result["fleet"] >= result["release"]
    assert result["annual_mileage_km"] > 0
    assert result["annual_in_service_hours"] > 0

    costs = result["economics"]
    components = (
        "fuel_energy_mln",
        "repair_mln",
        "crew_mln",
        "infrastructure_mln",
        "dispatch_mln",
        "contract_mln",
        "amortization_mln",
    )
    assert all(math.isfinite(costs[name]) and costs[name] >= 0 for name in components)
    assert costs["total_annual_mln"] == pytest.approx(sum(costs[name] for name in components))
    assert result["annual_total_cost_mln"] == pytest.approx(costs["total_annual_mln"])
    assert result["cost_per_km_rub"] == pytest.approx(
        costs["total_annual_mln"] * 1_000_000 / result["annual_mileage_km"]
    )


def test_economics_changes_with_route_length():
    short = calculate_annual_route_economics(
        vehicle_type="admiral_bk", route_length_km=10.0, max_section_flow_pph=500.0
    )
    long = calculate_annual_route_economics(
        vehicle_type="admiral_bk", route_length_km=20.0, max_section_flow_pph=500.0
    )
    assert long["annual_mileage_km"] > short["annual_mileage_km"]
    assert long["economics"]["infrastructure_mln"] >= short["economics"]["infrastructure_mln"]
    assert long["annual_total_cost_mln"] > short["annual_total_cost_mln"]


def test_economics_uses_peak_release_for_fleet_not_sum_of_periods():
    # A profile changes service intensity but fleet-fixed costs are not charged six times.
    base = calculate_annual_route_economics(
        vehicle_type="admiral_bk", route_length_km=20.0, max_section_flow_pph=500.0
    )
    custom = calculate_annual_route_economics(
        vehicle_type="admiral_bk", route_length_km=20.0, max_section_flow_pph=500.0,
        frequency_profile=((1.0, 0.5), (2.0, 1.0), (7.5, 0.5), (3.0, 1.0), (1.5, 0.5), (3.0, 0.5)),
    )
    assert base["fleet"] == custom["fleet"]
    assert base["economics"]["contract_mln"] == custom["economics"]["contract_mln"]
    assert base["economics"]["amortization_mln"] == custom["economics"]["amortization_mln"]
