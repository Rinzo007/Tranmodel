"""Cache-aware helpers for six-period AequilibraE evaluation.

Only JSON-serializable evaluation results are cached. AequilibraE/SQLite
objects are deliberately never persisted by this layer.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable

from .aequilibrae_eval_cache import load_json, save_json, stable_route_set_key


def period_cache_key(route_set, config, period_id: str, demand_factor: float, frequency_factor: float) -> str:
    return stable_route_set_key(
        route_set,
        config,
        extra={
            "period_id": str(period_id),
            "demand_factor": float(demand_factor),
            "frequency_factor": float(frequency_factor),
        },
    )


def evaluate_cached_period(
    route_set,
    config,
    period_id: str,
    demand_factor: float,
    frequency_factor: float,
    cache_dir: Path | str | None,
    compute: Callable[[], Any],
) -> tuple[Any, bool, str]:
    """Load a cached JSON evaluation or compute and cache it.

    Returns ``(value, hit, key)``. Objects that cannot be represented by JSON
    are simply returned from ``compute`` and are not cached.
    """
    if cache_dir is None:
        return compute(), False, ""
    key = period_cache_key(route_set, config, period_id, demand_factor, frequency_factor)
    root = Path(cache_dir)
    cached = load_json(root, key)
    if cached is not None:
        return cached, True, key
    value = compute()
    if isinstance(value, dict):
        try:
            save_json(root, key, value)
        except (OSError, TypeError, ValueError):
            pass
    return value, False, key


def cache_stats(cache_dir: Path | str | None) -> dict[str, int]:
    if cache_dir is None:
        return {"entries": 0, "bytes": 0}
    root = Path(cache_dir)
    if not root.exists():
        return {"entries": 0, "bytes": 0}
    files = list(root.glob("*.json"))
    return {"entries": len(files), "bytes": sum(p.stat().st_size for p in files if p.exists())}
