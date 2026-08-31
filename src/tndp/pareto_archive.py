"""Bounded Pareto archive used by the TNDP search."""
from __future__ import annotations
from dataclasses import dataclass, field
from .pareto import DEFAULT_OBJECTIVES, pareto_front

@dataclass
class ParetoArchive:
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES
    max_size: int = 100
    items: list[dict] = field(default_factory=list)

    def add(self, item: dict) -> bool:
        candidate = dict(item)
        pool = self.items + [candidate]
        front = pareto_front(pool, self.objectives)
        changed = candidate in front and candidate not in self.items
        if len(front) > self.max_size:
            front.sort(key=lambda x: (float(x.get("score", 0.0)), float(x.get("annual_cost_mln", 0.0))))
            front = front[: self.max_size]
        self.items = front
        return changed

    def extend(self, items: list[dict]) -> int:
        added = 0
        for item in items:
            added += int(self.add(item))
        return added

    def best(self, key: str = "score") -> dict | None:
        return min(self.items, key=lambda x: float(x.get(key, float("inf")))) if self.items else None

    def as_dict(self) -> dict:
        return {"objectives": list(self.objectives), "size": len(self.items), "items": self.items}
