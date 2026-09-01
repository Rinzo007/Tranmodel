"""Reusable acceleration primitives for TNDP search.

The optimizer can evaluate thousands of neighboring networks. This module keeps
candidate screening cheap and deterministic: duplicate networks are collapsed,
fast scores are memoized, and only the best K candidates are sent to the full
assignment evaluator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

@dataclass
class MemoizedEvaluator:
    evaluator: Callable[[T], R]
    key_fn: Callable[[T], Hashable]
    cache: dict[Hashable, R] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def __call__(self, value: T) -> R:
        key = self.key_fn(value)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        result = self.evaluator(value)
        self.cache[key] = result
        return result

    def clear(self) -> None:
        self.cache.clear()
        self.hits = self.misses = 0


def unique_by_key(values: Iterable[T], key_fn: Callable[[T], Hashable]) -> list[T]:
    """Remove duplicate candidates while preserving first-seen order."""
    seen: set[Hashable] = set()
    result: list[T] = []
    for value in values:
        key = key_fn(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def screen_then_exact(
    candidates: Iterable[T],
    *,
    key_fn: Callable[[T], Hashable],
    fast_evaluator: Callable[[T], float],
    exact_evaluator: Callable[[T], R],
    top_k: int,
) -> list[tuple[T, R]]:
    """Deduplicate, rank with a cheap score, then exactly evaluate top K."""
    if top_k < 1:
        return []
    unique = unique_by_key(candidates, key_fn)
    fast = MemoizedEvaluator(fast_evaluator, key_fn)
    ranked = sorted((fast(item), item) for item in unique)
    exact = [(item, exact_evaluator(item)) for _, item in ranked[:top_k]]
    return exact


def cache_stats(evaluator: MemoizedEvaluator) -> dict[str, int]:
    return {"hits": evaluator.hits, "misses": evaluator.misses, "size": len(evaluator.cache)}
