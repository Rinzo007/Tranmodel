from pathlib import Path
from types import SimpleNamespace

from src.tndp.model import Route, RouteSet
from src.tndp.multiperiod_cache import evaluate_cached_period, period_cache_key, cache_stats


def _config():
    return SimpleNamespace(name="test-config", speed_kmh=18.0)


def test_period_cache_distinguishes_period_and_factors(tmp_path: Path):
    routes = RouteSet([Route((1, 2), route_id="r1")])
    base = period_cache_key(routes, _config(), "1", 0.8, 0.8)
    other_period = period_cache_key(routes, _config(), "2", 1.0, 1.0)
    other_demand = period_cache_key(routes, _config(), "1", 1.0, 0.8)
    assert base != other_period
    assert base != other_demand


def test_period_cache_roundtrip(tmp_path: Path):
    routes = RouteSet([Route((1, 2), route_id="r1")])
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"evaluation": {"score": 123.0}}

    first, hit1, _ = evaluate_cached_period(routes, _config(), "1", 0.8, 0.8, tmp_path, compute)
    second, hit2, _ = evaluate_cached_period(routes, _config(), "1", 0.8, 0.8, tmp_path, compute)
    assert first == second
    assert hit1 is False
    assert hit2 is True
    assert calls["n"] == 1
    assert cache_stats(tmp_path)["entries"] == 1
