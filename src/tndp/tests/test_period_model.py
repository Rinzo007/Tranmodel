from src.tndp.interval_profile import DEFAULT_INTERVAL_PROFILE, daily_frequency_factor, validate_profile
from src.tndp.period_assignment import build_period_demands
from src.tndp.period_costs import route_period_costs
import numpy as np


def test_interval_profile_total_is_14_5():
    validate_profile()
    assert abs(daily_frequency_factor() - 14.5) < 1e-9


def test_period_demands_have_six_periods():
    base = np.eye(3)
    periods = build_period_demands(base)
    assert len(periods) == 6
    assert periods[1].matrix.shape == (3, 3)


def test_route_period_costs_returns_annual_components():
    result = route_period_costs(route_length_km=10.0, peak_flow_pph=400.0,
                                vehicle_type="liaz")
    assert result["peak_fleet"] >= 1
    assert result["annual_mileage_km"] > 0
    assert result["fuel_energy_mln"] > 0
    assert result["repair_mln"] > 0
