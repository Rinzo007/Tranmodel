"""Pareto-front utilities for TNDP solution archives."""
from __future__ import annotations


def dominates(a: dict, b: dict, objectives: tuple[str, ...]) -> bool:
    """Return True when a is no worse in every objective and better in one.

    Objective values are assumed to be minimized. Coverage is represented by
    its deficit (1 - coverage), so all objectives remain minimization targets.
    """
    av = [float(a.get(k, 0.0) or 0.0) for k in objectives]
    bv = [float(b.get(k, 0.0) or 0.0) for k in objectives]
    return all(x <= y + 1e-12 for x, y in zip(av, bv)) and any(x < y - 1e-12 for x, y in zip(av, bv))


def pareto_front(items: list[dict], objectives: tuple[str, ...]) -> list[dict]:
    """Return nondominated items while preserving deterministic input order."""
    front: list[dict] = []
    for i, item in enumerate(items):
        if any(dominates(other, item, objectives) for j, other in enumerate(items) if i != j):
            continue
        front.append(item)
    return front


def compact_solution_record(*, score: float, route_count: int, annual_cost_mln: float,
                            uncovered_demand: float, coverage_share: float,
                            user_cost: float, transfers: float, fleet: int,
                            metadata: dict | None = None) -> dict:
    """Create a serializable multi-objective record for the solution archive."""
    return {
        "score": float(score),
        "route_count": int(route_count),
        "annual_cost_mln": float(annual_cost_mln),
        "uncovered_demand": float(uncovered_demand),
        "coverage_share": float(coverage_share),
        "coverage_deficit": float(max(0.0, 1.0 - coverage_share)),
        "user_cost": float(user_cost),
        "transfers": float(transfers),
        "fleet": int(fleet),
        "metadata": metadata or {},
    }


DEFAULT_OBJECTIVES = (
    "user_cost",
    "transfers",
    "uncovered_demand",
    "coverage_deficit",
    "annual_cost_mln",
)
