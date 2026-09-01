"""Small, deterministic caches for expensive TNDP evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Hashable


@dataclass
class EvaluationCache:
    """Memoize pure candidate evaluations and expose hit/miss counters."""
    _values: dict[Hashable, Any] = None  # type: ignore[assignment]
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        if self._values is None:
            self._values = {}

    @staticmethod
    def key(payload: Any) -> str:
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(normalized.encode("utf-8")).hexdigest()

    def get_or_compute(self, payload: Any, compute: Callable[[], Any]) -> Any:
        key = self.key(payload)
        if key in self._values:
            self.hits += 1
            return self._values[key]
        self.misses += 1
        value = compute()
        self._values[key] = value
        return value

    def clear(self) -> None:
        self._values.clear()
        self.hits = self.misses = 0


@dataclass
class RouteEvaluationCache:
    """Cache route-level service/economic calculations independently of assignment."""
    service: EvaluationCache = None  # type: ignore[assignment]
    economics: EvaluationCache = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.service is None:
            self.service = EvaluationCache()
        if self.economics is None:
            self.economics = EvaluationCache()

    def clear(self) -> None:
        self.service.clear()
        self.economics.clear()
