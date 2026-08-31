"""Rolling-stock capacities and operating calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class VehicleType:
    code: str
    name: str
    capacity: float
    technical_readiness: float
    electric: bool = False


VEHICLE_TYPES: dict[str, VehicleType] = {
    "paz_3205": VehicleType("paz_3205", "МК ПАЗ 3205", 18, 0.80),
    "liaz_4292": VehicleType("liaz_4292", "СК ЛИАЗ 4292", 43, 0.80),
    "liaz_5292": VehicleType("liaz_5292", "БК ЛИАЗ 5292", 68, 0.80),
    "electric_liaz": VehicleType("electric_liaz", "Эл.БК ЛиАЗ", 73, 0.80, True),
    "electric_kamaz": VehicleType("electric_kamaz", "Эл.БК КамАЗ", 72, 0.80, True),
    "ziu_9": VehicleType("ziu_9", "Тб БК ЗИУ-9", 73, 0.90),
    "liaz_6213": VehicleType("liaz_6213", "ОБК ЛИАЗ 6213", 93, 0.80),
    "ziu_10": VehicleType("ziu_10", "Тб ОБК ЗИУ-10", 98, 0.90),
    "tm_71_911": VehicleType("tm_71_911", "Тм-МК 71-911ЕМ", 95, 0.90),
    "tm_bogatyr": VehicleType("tm_bogatyr", "Тм-СК Богатырь", 111, 0.90),
    "tm_vityaz": VehicleType("tm_vityaz", "Тм-БК Витязь", 162, 0.90),
    "tm_lev": VehicleType("tm_lev", "Тм-ОБК Лев", 226, 0.90),
}

DEFAULT_VEHICLE_TYPE = "electric_liaz"


def get_vehicle_type(code: str) -> VehicleType:
    try:
        return VEHICLE_TYPES[code]
    except KeyError as exc:
        raise ValueError(f"Unknown vehicle type: {code}") from exc


def round_down_half_minutes(minutes: float) -> float:
    return math.floor(minutes * 2.0 + 1e-9) / 2.0


def round_up_to_interval(minutes: float, interval_min: float) -> float:
    if interval_min <= 0:
        raise ValueError("interval_min must be positive")
    return math.ceil(minutes / interval_min - 1e-12) * interval_min


def calculate_route_operations(
    *,
    route_length_km: float,
    max_section_flow_pph: float,
    vehicle_type: str = DEFAULT_VEHICLE_TYPE,
    speed_kmh: float = 18.0,
    interval_reserve_sec: float = 20.0,
    terminal_delay_reserve: float = 0.08,
    charging_min_per_terminal: float = 10.0,
    annual_days: int = 350,
    park_trip_coefficient: float = 0.90,
    frequency_profile: tuple[tuple[float, float], ...] = ((3.0, 1.0), (6.0, 0.75), (4.0, 1.0), (3.0, 0.60), (8.0, 0.30)),
) -> dict[str, float | str]:
    """Calculate the requested conditional-route operating indicators."""
    if route_length_km <= 0 or max_section_flow_pph < 0 or speed_kmh <= 0:
        raise ValueError("Invalid route operating inputs")
    vehicle = get_vehicle_type(vehicle_type)
    frequency_vph = max_section_flow_pph / vehicle.capacity if max_section_flow_pph > 0 else 0.0
    # No demand means the minimum useful service frequency is applied.
    frequency_vph = max(frequency_vph, 0.1)
    raw_interval = 60.0 / frequency_vph
    interval_min = round_down_half_minutes(raw_interval + interval_reserve_sec / 60.0)
    interval_min = max(interval_min, 0.5)
    frequency_vph = 60.0 / interval_min

    running_min = route_length_km / speed_kmh * 60.0
    cycle_min = running_min * (1.0 + terminal_delay_reserve) + 2.0 * charging_min_per_terminal if vehicle.electric else running_min * (1.0 + terminal_delay_reserve)
    cycle_min = round_up_to_interval(cycle_min, interval_min)
    release = cycle_min / interval_min
    fleet = math.ceil(release / vehicle.technical_readiness - 1e-9)

    # frequency_profile is (hours in period, multiplier). The multiplier scales
    # the peak frequency. This yields the number of scheduled round trips/day.
    daily_trips = 0.0
    for hours, multiplier in frequency_profile:
        daily_trips += hours * frequency_vph * multiplier
    annual_mileage_km = route_length_km * daily_trips / park_trip_coefficient * annual_days
    annual_in_service_hours = (cycle_min / 60.0) * daily_trips / park_trip_coefficient * annual_days

    return {
        "vehicle_type": vehicle.code,
        "vehicle_name": vehicle.name,
        "capacity": vehicle.capacity,
        "max_section_flow_pph": float(max_section_flow_pph),
        "speed_kmh": speed_kmh,
        "frequency_vph": frequency_vph,
        "interval_min": interval_min,
        "terminal_delay_reserve": terminal_delay_reserve,
        "charging_min_per_terminal": charging_min_per_terminal if vehicle.electric else 0.0,
        "cycle_time_min": cycle_min,
        "release": release,
        "technical_readiness": vehicle.technical_readiness,
        "fleet": float(fleet),
        "daily_trips": daily_trips,
        "annual_mileage_km": annual_mileage_km,
        "annual_in_service_hours": annual_in_service_hours,
    }
