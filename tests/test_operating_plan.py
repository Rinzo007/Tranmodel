from __future__ import annotations

from src.tndp.interval_profile import DEFAULT_INTERVAL_PROFILE, daily_frequency_factor
from src.tndp.peak_fleet import reconcile_route_periods
from src.tndp.period_vehicle_plan import build_route_vehicle_plan_auto


def test_interval_profile_total_is_14_5():
    assert daily_frequency_factor() == 14.5
    assert len(DEFAULT_INTERVAL_PROFILE) == 6


def test_peak_fleet_is_maximum_across_periods():
    result = reconcile_route_periods(
        route_length_km=10.0,
        period_peak_flows=[300, 500, 350, 600, 320, 180],
        vehicle_type="liaz",
    )
    fleets = [p["fleet"] for p in result["periods"]]
    assert result["peak_fleet"] == max(fleets)
    assert len(fleets) == 6


def test_auto_vehicle_uses_worst_period():
    result = build_route_vehicle_plan_auto(
        route_id="R1",
        route_length_km=10.0,
        period_peak_flows=[250, 1200, 400, 900, 300, 150],
        allowed_vehicle_types=("liaz", "liaz_obk", "tm_lvenok", "tm_vityaz"),
    )
    assert result["peak_fleet"] >= 1
    assert result["periods"]
    assert result["costs"]["total_annual_mln"] >= 0
