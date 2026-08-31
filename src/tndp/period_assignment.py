"""Period-dependent demand scaling and assignment-plan aggregation.

This module deliberately keeps the AequilibraE call outside the period model:
it produces the six demand matrices/factors and a common structure for storing
assignment outputs. The runner can execute one assignment per period.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence
import numpy as np
from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod

@dataclass(frozen=True, slots=True)
class PeriodDemand:
    period: str
    start: str
    end: str
    hours: float
    frequency_factor: float
    demand_factor: float
    matrix: np.ndarray

@dataclass(frozen=True, slots=True)
class PeriodAssignment:
    period: str
    user_cost: float = 0.0
    waiting_time: float = 0.0
    walking_time: float = 0.0
    transfers: float = 0.0
    uncovered_demand: float = 0.0
    capacity_excess: float = 0.0
    annualized_weight: float = 0.0


def build_period_demands(base_demand: np.ndarray, *, demand_factors: Mapping[str, float] | None = None,
                         profile: Sequence[IntervalPeriod] = DEFAULT_INTERVAL_PROFILE) -> list[PeriodDemand]:
    """Create one OD matrix per operating period.

    Factors are normalized so the supplied factors represent relative period
    demand. With no custom mapping, the frequency profile is also used as the
    initial demand proxy; this is explicit and can later be replaced by a
    calibrated time-of-day demand profile.
    """
    base = np.asarray(base_demand, dtype=float)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("base_demand must be a square OD matrix")
    out: list[PeriodDemand] = []
    for p in profile:
        factor = float((demand_factors or {}).get(p.name, p.frequency_factor))
        if factor < 0:
            raise ValueError(f"Negative demand factor for period {p.name}")
        out.append(PeriodDemand(p.name, p.start, p.end, p.hours, p.frequency_factor, factor, base * factor))
    return out


def aggregate_period_assignments(results: Sequence[PeriodAssignment]) -> dict:
    if not results:
        return {"periods": [], "user_cost": 0.0, "waiting_time": 0.0, "walking_time": 0.0,
                "transfers": 0.0, "uncovered_demand": 0.0, "capacity_excess": 0.0}
    weight = sum(max(r.annualized_weight, 0.0) for r in results) or 1.0
    def weighted(name: str) -> float:
        return sum(float(getattr(r, name)) * max(r.annualized_weight, 0.0) for r in results) / weight
    return {
        "periods": [asdict(r) for r in results],
        "user_cost": weighted("user_cost"),
        "waiting_time": weighted("waiting_time"),
        "walking_time": weighted("walking_time"),
        "transfers": weighted("transfers"),
        "uncovered_demand": sum(float(r.uncovered_demand) for r in results),
        "capacity_excess": max(float(r.capacity_excess) for r in results),
    }
