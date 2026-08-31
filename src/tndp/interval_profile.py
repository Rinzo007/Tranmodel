"""Canonical daily service-frequency profile for route operating calculations."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class IntervalPeriod:
    number: int
    name: str
    start: str
    end: str
    frequency_factor: float
    hours: float
    @property
    def trips_factor(self) -> float:
        return self.hours * self.frequency_factor

DEFAULT_INTERVAL_PROFILE = (
    IntervalPeriod(1, "Межпик", "06:00", "07:00", 0.8, 1.0),
    IntervalPeriod(2, "Пик", "07:00", "09:00", 1.0, 2.0),
    IntervalPeriod(3, "Межпик", "09:00", "16:30", 0.8, 7.5),
    IntervalPeriod(4, "Пик", "16:30", "19:30", 1.0, 3.0),
    IntervalPeriod(5, "Межпик", "19:30", "21:00", 0.8, 1.5),
    IntervalPeriod(6, "Вечер", "21:00", "00:00", 0.5, 3.0),
)

def daily_frequency_factor(profile=DEFAULT_INTERVAL_PROFILE) -> float:
    return sum(p.trips_factor for p in profile)

def as_frequency_profile(profile=DEFAULT_INTERVAL_PROFILE) -> tuple[tuple[float, float], ...]:
    """Compatibility representation used by the operating/economic calculators."""
    return tuple((p.hours, p.frequency_factor) for p in profile)

def validate_profile(profile=DEFAULT_INTERVAL_PROFILE) -> None:
    if not profile: raise ValueError("Interval profile cannot be empty")
    if abs(daily_frequency_factor(profile) - 14.5) > 1e-9:
        raise ValueError(f"Interval profile must total 14.5 peak-frequency hours, got {daily_frequency_factor(profile)}")
    previous_end = None
    for p in profile:
        if p.hours <= 0 or not 0 < p.frequency_factor <= 1: raise ValueError(f"Invalid interval period: {p.name}")
        if previous_end is not None and p.start != previous_end: raise ValueError("Interval profile contains a time gap or overlap")
        previous_end = p.end
    if profile[-1].end != "00:00": raise ValueError("Interval profile must end at 00:00")
