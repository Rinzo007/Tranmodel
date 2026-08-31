"""Rolling-stock catalogue and route operating calculations.

The catalogue follows the user's 16-position operating/economic table.
Monetary values are million currency units; capacities are planning passenger
capacities used for frequency calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class VehicleType:
    code: str
    name: str
    mode: str
    capacity_class: str
    capacity: float
    unit_cost_mln: float
    major_repair_share: float
    contract_months: int
    service_life_years: int
    annual_contract_cost_mln: float
    annual_amortization_mln: float
    one_off_cost_mln: float
    technical_readiness: float
    electric: bool = False
    charging_at_terminal: bool = False


VEHICLE_TYPES: dict[str, VehicleType] = {
    # Автобусы
    "ford_transit": VehicleType("ford_transit", "Форд-Транзит", "Авт", "МК", 18, 2.60, 0.00, 12, 5, 0.52, 74, 371.8, 0.80),
    "gazelle_city": VehicleType("gazelle_city", "Газель-Сити", "Авт", "МК", 18, 3.20, 0.00, 12, 5, 0.64, 69, 345.6, 0.80),
    "paz": VehicleType("paz", "ПАЗ", "Авт", "СК", 43, 4.10, 0.00, 12, 5, 0.82, 39, 192.7, 0.80),
    "liaz": VehicleType("liaz", "ЛИАЗ", "Авт", "БК", 68, 12.00, 0.00, 12, 7, 1.71, 55, 384.0, 0.80),
    "liaz_gas": VehicleType("liaz_gas", "ЛИАЗ, газовый", "Авт", "БК", 68, 12.00, 0.00, 12, 7, 1.71, 55, 384.0, 0.80),
    "liaz_obk": VehicleType("liaz_obk", "Лиаз ОБК", "Авт", "ОБК", 93, 16.00, 0.00, 12, 7, 2.29, 50, 352.0, 0.80),
    "liaz_obk_gas": VehicleType("liaz_obk_gas", "Лиаз ОБК, газовый", "Авт", "ОБК", 93, 16.00, 0.00, 12, 7, 2.29, 50, 352.0, 0.80),
    # Электробус
    "kamaz_charge_terminal": VehicleType("kamaz_charge_terminal", "Камаз, зарядка на конечной", "Элб", "БК", 72, 34.40, 0.30, 12, 15, 2.98, 98, 1135.0, 0.80, True, True),
    # Троллейбусы
    "admiral_bk": VehicleType("admiral_bk", "Адмирал", "Тб", "БК", 73, 22.00, 0.00, 12, 15, 1.47, 40, 594.0, 0.90),
    "admiral_obk": VehicleType("admiral_obk", "Адмирал ОБК", "Тб", "ОБК", 98, 28.00, 0.00, 12, 15, 1.87, 41, 616.0, 0.90),
    # ТУАХ — автономный ход. В таблице приведены два класса вместимости.
    "tuah_bk": VehicleType("tuah_bk", "БК", "ТУАХ", "БК", 73, 27.00, 0.30, 12, 15, 2.34, 63, 729.0, 0.90, True, True),
    "tuah_obk": VehicleType("tuah_obk", "Адмирал ОБК", "ТУАХ", "ОБК", 98, 33.00, 0.30, 12, 15, 2.86, 63, 726.0, 0.90, True, True),
    # Трамваи. Для сцепок вместимость масштабируется по числу секций/единиц.
    "tm_lvenok": VehicleType("tm_lvenok", "Львенок", "Тм", "БК", 95, 45.00, 0.38, 12, 30, 2.07, 46, 990.0, 0.90),
    "tm_vityaz": VehicleType("tm_vityaz", "Витязь", "Тм", "ОБК", 162, 96.00, 0.38, 12, 30, 4.42, 57, 1248.0, 0.90),
    "tm_2x_bk": VehicleType("tm_2x_bk", "2хБК", "Тм", "2хБК", 190, 90.00, 0.38, 12, 30, 4.14, 54, 1170.0, 0.90),
    "tm_3x_bk": VehicleType("tm_3x_bk", "3хБК", "Тм", "3хБК", 285, 135.00, 0.38, 12, 30, 6.21, 81, 1755.0, 0.90),
}

DEFAULT_VEHICLE_TYPE = "kamaz_charge_terminal"


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
    """Calculate operating and lifecycle indicators for one route."""
    if route_length_km <= 0 or max_section_flow_pph < 0 or speed_kmh <= 0:
        raise ValueError("Invalid route operating inputs")

    vehicle = get_vehicle_type(vehicle_type)
    frequency_vph = max(max_section_flow_pph / vehicle.capacity, 0.1)
    raw_interval = 60.0 / frequency_vph
    interval_min = max(round_down_half_minutes(raw_interval + interval_reserve_sec / 60.0), 0.5)
    frequency_vph = 60.0 / interval_min

    running_min = route_length_km / speed_kmh * 60.0
    cycle_min = running_min * (1.0 + terminal_delay_reserve)
    if vehicle.charging_at_terminal:
        cycle_min += 2.0 * charging_min_per_terminal
    cycle_min = round_up_to_interval(cycle_min, interval_min)

    release = cycle_min / interval_min
    fleet = math.ceil(release / vehicle.technical_readiness - 1e-9)
    daily_trips = sum(hours * frequency_vph * multiplier for hours, multiplier in frequency_profile)
    annual_mileage_km = route_length_km * daily_trips / park_trip_coefficient * annual_days
    annual_in_service_hours = cycle_min / 60.0 * daily_trips / park_trip_coefficient * annual_days

    return {
        "vehicle_type": vehicle.code,
        "vehicle_name": vehicle.name,
        "mode": vehicle.mode,
        "capacity_class": vehicle.capacity_class,
        "capacity": vehicle.capacity,
        "unit_cost_mln": vehicle.unit_cost_mln,
        "major_repair_share": vehicle.major_repair_share,
        "contract_months": vehicle.contract_months,
        "service_life_years": vehicle.service_life_years,
        "annual_contract_cost_mln": vehicle.annual_contract_cost_mln,
        "annual_amortization_mln": vehicle.annual_amortization_mln,
        "one_off_cost_mln": vehicle.one_off_cost_mln,
        "max_section_flow_pph": float(max_section_flow_pph),
        "speed_kmh": speed_kmh,
        "frequency_vph": frequency_vph,
        "interval_min": interval_min,
        "terminal_delay_reserve": terminal_delay_reserve,
        "charging_min_per_terminal": charging_min_per_terminal if vehicle.charging_at_terminal else 0.0,
        "cycle_time_min": cycle_min,
        "release": release,
        "technical_readiness": vehicle.technical_readiness,
        "fleet": float(fleet),
        "daily_trips": daily_trips,
        "annual_mileage_km": annual_mileage_km,
        "annual_in_service_hours": annual_in_service_hours,
        "annual_fleet_contract_cost_mln": fleet * vehicle.annual_contract_cost_mln,
        "annual_fleet_amortization_mln": fleet * vehicle.annual_amortization_mln,
        "one_off_fleet_cost_mln": fleet * vehicle.one_off_cost_mln,
    }
