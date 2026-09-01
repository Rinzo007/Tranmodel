"""Population-weighted transit accessibility metrics for TNDP."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .model import RouteSet


def population_coverage(
    route_set: RouteSet,
    zone_xy_m: np.ndarray,
    population: np.ndarray,
    stop_xy_m: np.ndarray,
    radii_m: tuple[float, ...] = (400.0, 500.0, 800.0),
) -> dict[str, float]:
    """Calculate population-weighted coverage by the nearest stop used by a route.

    Distances are straight-line distances in the projected coordinate system.
    This is intentionally cheap enough to run during TNDP screening; the exact
    pedestrian-network calculation can replace it later without changing the API.
    """
    zone_xy_m = np.asarray(zone_xy_m, dtype=float)
    population = np.asarray(population, dtype=float)
    stop_xy_m = np.asarray(stop_xy_m, dtype=float)
    total = float(np.clip(population, 0.0, None).sum())
    if total <= 0 or len(zone_xy_m) == 0 or len(stop_xy_m) == 0:
        return {f"coverage_{int(r)}m": 0.0 for r in radii_m} | {"coverage_population": 0.0}

    used = sorted({int(n) for route in route_set.routes for n in route.nodes if 0 <= int(n) < len(stop_xy_m)})
    if not used:
        return {f"coverage_{int(r)}m": 0.0 for r in radii_m} | {"coverage_population": 0.0}

    tree = cKDTree(stop_xy_m[np.asarray(used, dtype=int)])
    distance_m, _ = tree.query(zone_xy_m, k=1)
    pop = np.clip(population, 0.0, None)
    result = {f"coverage_{int(r)}m": float(pop[distance_m <= float(r)].sum() / total) for r in radii_m}
    result["coverage_population"] = total
    result["coverage_400_800_gap"] = max(0.0, result.get("coverage_800m", 0.0) - result.get("coverage_400m", 0.0))
    return result
