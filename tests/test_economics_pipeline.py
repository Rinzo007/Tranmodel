from src.tndp.interval_profile import DEFAULT_INTERVAL_PROFILE, daily_frequency_factor
from src.tndp.cost_aggregation import aggregate_peak_fleet


def test_interval_profile_total_is_14_5():
    assert abs(daily_frequency_factor(DEFAULT_INTERVAL_PROFILE) - 14.5) < 1e-9


def test_peak_fleet_uses_simultaneous_maximum():
    assert aggregate_peak_fleet([3, 7, 5, 4, 6, 2]) == 7
