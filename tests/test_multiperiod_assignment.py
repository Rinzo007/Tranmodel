import pytest

import numpy as np

from src.tndp.interval_profile import DEFAULT_INTERVAL_PROFILE, daily_frequency_factor
from src.tndp.multiperiod_assignment import _period_route_set
from src.tndp.model import Route, RouteSet


def test_interval_profile_has_six_periods_and_14_5_factor():
    assert len(DEFAULT_INTERVAL_PROFILE) == 6
    assert daily_frequency_factor() == 14.5


def test_period_route_set_preserves_route_metadata():
    route = Route((1, 2, 3), "R1", 6.0, 420.0, "liaz")
    scaled = _period_route_set(RouteSet([route]), 0.8).routes[0]
    assert scaled.frequency_vph == pytest.approx(4.8)
    assert scaled.max_section_flow_pph == 420.0
    assert scaled.vehicle_type == "liaz"
    assert scaled.route_id == "R1"
