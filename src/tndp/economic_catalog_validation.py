"""Consistency checks for the 16-vehicle economic catalogue."""
from __future__ import annotations
from .vehicle_types import VEHICLE_TYPES
from .operating_costs import (
    FUEL_ENERGY_RUB_PER_KM, REPAIR_RUB_PER_KM, INFRASTRUCTURE_ANNUAL_MLN,
    DISPATCH_ANNUAL_MLN, DRIVER_SALARY_COEFFICIENT, DRIVER_HOUR_RUB,
    DRIVER_HOUR_WITH_CHARGES_RUB, REPAIR_WORKER_SALARY_COEFFICIENT,
    TO_LABOUR_HOURS_PER_1000KM, SPARE_PARTS_RUB_PER_KM, TYRE_RUB_PER_KM,
)

_REQUIRED = {
    "fuel_energy": FUEL_ENERGY_RUB_PER_KM,
    "repair": REPAIR_RUB_PER_KM,
    "infrastructure": INFRASTRUCTURE_ANNUAL_MLN,
    "dispatch": DISPATCH_ANNUAL_MLN,
    "driver_salary": DRIVER_SALARY_COEFFICIENT,
    "driver_hour": DRIVER_HOUR_RUB,
    "driver_hour_with_charges": DRIVER_HOUR_WITH_CHARGES_RUB,
    "repair_salary": REPAIR_WORKER_SALARY_COEFFICIENT,
    "to_labour": TO_LABOUR_HOURS_PER_1000KM,
    "spare_parts": SPARE_PARTS_RUB_PER_KM,
    "tyres": TYRE_RUB_PER_KM,
}


def validate_economic_catalogue() -> dict:
    """Return a machine-readable audit report; raise no exception by default."""
    vehicle_codes = set(VEHICLE_TYPES)
    missing: dict[str, list[str]] = {}
    extra: dict[str, list[str]] = {}
    for name, mapping in _REQUIRED.items():
        keys = set(mapping)
        miss = sorted(vehicle_codes - keys)
        ext = sorted(keys - vehicle_codes)
        if miss: missing[name] = miss
        if ext: extra[name] = ext
    invalid_values = []
    for code, v in VEHICLE_TYPES.items():
        for field in ("capacity", "unit_cost_mln", "annual_contract_cost_mln", "annual_amortization_mln", "one_off_cost_mln"):
            if getattr(v, field) < 0:
                invalid_values.append(f"{code}.{field}<0")
        if not 0 < v.technical_readiness <= 1:
            invalid_values.append(f"{code}.technical_readiness outside (0,1]")
    ok = not missing and not extra and not invalid_values and len(vehicle_codes) == 16
    return {"ok": ok, "vehicle_count": len(vehicle_codes), "missing": missing,
            "extra": extra, "invalid_values": invalid_values, "vehicle_codes": sorted(vehicle_codes)}


def assert_economic_catalogue() -> None:
    report = validate_economic_catalogue()
    assert report["ok"], report
