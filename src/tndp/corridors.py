"""Extract spatial demand corridors from an OD matrix."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True, slots=True)
class DemandCorridor:
    origin: int
    destination: int
    demand: float
    direct_distance_km: float


def extract_demand_corridors(
    demand: np.ndarray,
    node_xy_km: np.ndarray,
    top_pairs: int = 300,
    max_distance_km: float = 8.0,
    min_demand: float = 0.0,
) -> list[DemandCorridor]:
    """Return strongest OD pairs as corridor seeds.

    Pairs are directional. Near-duplicate reverse pairs are retained because
    public transport may need different directional service patterns.
    """
    matrix = np.asarray(demand, dtype=float)
    xy = np.asarray(node_xy_km, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("demand must be a square matrix")
    if xy.shape != (matrix.shape[0], 2):
        raise ValueError("node_xy_km must have shape (n, 2)")

    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    ii, jj = np.where(mask & (matrix > min_demand))
    if len(ii) == 0:
        return []
    distances = np.sqrt(((xy[ii] - xy[jj]) ** 2).sum(axis=1))
    keep = distances <= max_distance_km
    records = [
        DemandCorridor(int(i), int(j), float(matrix[i, j]), float(d))
        for i, j, d in zip(ii[keep], jj[keep], distances[keep])
    ]
    records.sort(key=lambda x: x.demand, reverse=True)
    return records[:top_pairs]


def cluster_corridors(
    corridors: list[DemandCorridor],
    node_xy_km: np.ndarray,
    radius_km: float = 1.0,
) -> list[list[DemandCorridor]]:
    """Group OD corridors with nearby origins and destinations."""
    if not corridors:
        return []
    xy = np.asarray(node_xy_km, dtype=float)
    result: list[list[DemandCorridor]] = []
    centers: list[tuple[int, int]] = []
    tree = cKDTree(xy)
    for corridor in corridors:
        matches = [k for k, (oi, di) in enumerate(centers)
                   if tree.query_ball_point(xy[corridor.origin], radius_km)
                   and tree.query_ball_point(xy[corridor.destination], radius_km)]
        if matches:
            result[matches[0]].append(corridor)
        else:
            centers.append((corridor.origin, corridor.destination))
            result.append([corridor])
    return result
