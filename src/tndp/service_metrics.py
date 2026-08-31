"""Service-quality metrics independent of the optimizer implementation."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ServiceMetrics:
    user_time_min: float = 0.0
    wait_time_min: float = 0.0
    walk_time_min: float = 0.0
    transfers: float = 0.0
    uncovered_demand: float = 0.0
    overload: float = 0.0
    direct_demand_share: float = 0.0

    def as_dict(self) -> dict:
        return {
            "user_time_min": float(self.user_time_min),
            "wait_time_min": float(self.wait_time_min),
            "walk_time_min": float(self.walk_time_min),
            "transfers": float(self.transfers),
            "uncovered_demand": float(self.uncovered_demand),
            "overload": float(self.overload),
            "direct_demand_share": float(self.direct_demand_share),
        }

def generalized_cost(m: ServiceMetrics, *, wait_weight: float = 1.5,
                     walk_weight: float = 2.0, transfer_weight: float = 1.0,
                     transfer_penalty_min: float = 8.0) -> float:
    """Return generalized passenger cost in equivalent minutes."""
    return (m.user_time_min + wait_weight * m.wait_time_min +
            walk_weight * m.walk_time_min +
            transfer_weight * m.transfers * transfer_penalty_min)
