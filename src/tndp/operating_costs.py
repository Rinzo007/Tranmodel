"""Operating cost coefficients supplied for the Voronezh TNDP model.

All monetary values are in rubles unless explicitly marked as million rubles.
The tables are kept separate from the rolling-stock catalogue so source
coefficients can be audited without changing route-generation logic.
"""
from __future__ import annotations

# Fuel / energy cost per km. Values already include the supplied network-loss /
# lubricant coefficient and passenger-compartment heating allowances.
FUEL_ENERGY_RUB_PER_KM = {
    "ford_transit": 13.45, "gazelle_city": 13.45, "paz": 28.33,
    "liaz": 42.43, "liaz_gas": 22.08, "liaz_obk": 49.25,
    "liaz_obk_gas": 25.85, "kamaz_charge_terminal": 10.80,
    "admiral_bk": 12.74, "admiral_obk": 19.65, "tuah_bk": 12.74,
    "tuah_obk": 19.65, "tm_lvenok": 14.34, "tm_vityaz": 24.43,
    "tm_2x_bk": 28.67, "tm_3x_bk": 43.01,
}

# Repair + spare parts + tyres, rub/km.
REPAIR_RUB_PER_KM = {
    "ford_transit": 7.12, "gazelle_city": 7.12, "paz": 8.58,
    "liaz": 13.18, "liaz_gas": 16.51, "liaz_obk": 17.71,
    "liaz_obk_gas": 21.07, "kamaz_charge_terminal": 20.46,
    "admiral_bk": 20.46, "admiral_obk": 25.47, "tuah_bk": 20.46,
    "tuah_obk": 25.47, "tm_lvenok": 22.69, "tm_vityaz": 29.39,
    "tm_2x_bk": 45.83, "tm_3x_bk": 68.97,
}

# Annual infrastructure cost, million rubles per route.
INFRASTRUCTURE_ANNUAL_MLN = {
    "ford_transit": 30.0, "gazelle_city": 30.0, "paz": 30.0,
    "liaz": 30.0, "liaz_gas": 30.0, "liaz_obk": 30.0,
    "liaz_obk_gas": 30.0, "kamaz_charge_terminal": 46.0,
    "admiral_bk": 34.0, "admiral_obk": 34.0, "tuah_bk": 33.0,
    "tuah_obk": 33.0, "tm_lvenok": 55.0, "tm_vityaz": 54.0,
    "tm_2x_bk": 54.0, "tm_3x_bk": 54.0,
}

# Annual dispatching cost, million rubles per route.
DISPATCH_ANNUAL_MLN = {
    "ford_transit": 6.1, "gazelle_city": 4.6, "paz": 2.0,
    "liaz": 1.3, "liaz_gas": 1.3, "liaz_obk": .9,
    "liaz_obk_gas": .9, "kamaz_charge_terminal": 1.4,
    "admiral_bk": 1.1, "admiral_obk": .9, "tuah_bk": 1.1,
    "tuah_obk": .9, "tm_lvenok": .9, "tm_vityaz": .5,
    "tm_2x_bk": .5, "tm_3x_bk": .5,
}

# Driver payroll parameters from the supplied table.
AVERAGE_MONTHLY_SALARY_RUB = 36_480.0
DRIVER_SALARY_COEFFICIENT = {
    "ford_transit": .98, "gazelle_city": .98, "paz": 1.05,
    "liaz": 1.43, "liaz_gas": 1.43, "liaz_obk": 1.50,
    "liaz_obk_gas": 1.50, "kamaz_charge_terminal": 1.00,
    "admiral_bk": 1.00, "admiral_obk": 1.10, "tuah_bk": 1.00,
    "tuah_obk": 1.10, "tm_lvenok": .90, "tm_vityaz": 1.00,
    "tm_2x_bk": 1.00, "tm_3x_bk": 1.10,
}
METROPOLIS_COEFFICIENT = 1.0
WORK_HOURS_YEAR = 1772.0
PREP_CLOSE_COEFFICIENT = 1.06
TICKET_SALES_COEFFICIENT = 1.0
SOCIAL_CONTRIBUTIONS = .308

# Supplied hourly rates after the salary coefficient calculation.
DRIVER_HOUR_RUB = {
    "ford_transit": 256.6, "gazelle_city": 256.6, "paz": 275.0,
    "liaz": 374.5, "liaz_gas": 374.5, "liaz_obk": 392.8,
    "liaz_obk_gas": 392.8, "kamaz_charge_terminal": 261.9,
    "admiral_bk": 261.9, "admiral_obk": 288.0, "tuah_bk": 261.9,
    "tuah_obk": 288.0, "tm_lvenok": 235.7, "tm_vityaz": 261.9,
    "tm_2x_bk": 261.9, "tm_3x_bk": 288.0,
}
DRIVER_HOUR_WITH_CHARGES_RUB = {
    "ford_transit": 335.7, "gazelle_city": 335.7, "paz": 359.6,
    "liaz": 489.8, "liaz_gas": 489.8, "liaz_obk": 513.8,
    "liaz_obk_gas": 513.8, "kamaz_charge_terminal": 342.5,
    "admiral_bk": 342.5, "admiral_obk": 376.8, "tuah_bk": 342.5,
    "tuah_obk": 376.8, "tm_lvenok": 308.3, "tm_vityaz": 342.5,
    "tm_2x_bk": 342.5, "tm_3x_bk": 376.8,
}

# Maintenance labour parameters.
REPAIR_WORKER_SALARY_COEFFICIENT = {
    "ford_transit": .80, "gazelle_city": .80, "paz": .80,
    "liaz": .80, "liaz_gas": .80, "liaz_obk": .80,
    "liaz_obk_gas": .80, "kamaz_charge_terminal": .90,
    "admiral_bk": .90, "admiral_obk": .90, "tuah_bk": .90,
    "tuah_obk": .90, "tm_lvenok": .90, "tm_vityaz": .90,
    "tm_2x_bk": .90, "tm_3x_bk": .90,
}
TO_LABOUR_HOURS_PER_1000KM = {
    "ford_transit": 8.00, "gazelle_city": 8.00, "paz": 9.30,
    "liaz": 13.30, "liaz_gas": 16.88, "liaz_obk": 19.10,
    "liaz_obk_gas": 22.93, "kamaz_charge_terminal": 25.00,
    "admiral_bk": 25.00, "admiral_obk": 30.00, "tuah_bk": 25.00,
    "tuah_obk": 30.00, "tm_lvenok": 20.00, "tm_vityaz": 26.00,
    "tm_2x_bk": 40.40, "tm_3x_bk": 60.80,
}
REPAIR_LABOUR_HOURS_PER_1000KM = {
    "ford_transit": 6.40, "gazelle_city": 6.40, "paz": 7.80,
    "liaz": 10.20, "liaz_gas": 12.95, "liaz_obk": 13.20,
    "liaz_obk_gas": 15.85,
}
SPARE_PARTS_RUB_PER_KM = {
    "ford_transit": 3.20, "gazelle_city": 3.20, "paz": 3.60,
    "liaz": 6.40, "liaz_gas": 8.12, "liaz_obk": 8.60,
    "liaz_obk_gas": 10.32, "kamaz_charge_terminal": 12.60,
    "admiral_bk": 12.60, "admiral_obk": 16.00, "tuah_bk": 12.60,
    "tuah_obk": 16.00, "tm_lvenok": 17.00, "tm_vityaz": 22.00,
    "tm_2x_bk": 34.34, "tm_3x_bk": 51.68,
}
TYRE_RUB_PER_KM = {
    "ford_transit": .28, "gazelle_city": .28, "paz": .66,
    "liaz": .84, "liaz_gas": .84, "liaz_obk": .94,
    "liaz_obk_gas": .94, "kamaz_charge_terminal": .75,
    "admiral_bk": .75, "admiral_obk": .94, "tuah_bk": .75,
    "tuah_obk": .94, "tm_lvenok": 0.0, "tm_vityaz": 0.0,
    "tm_2x_bk": 0.0, "tm_3x_bk": 0.0,
}

# Dispatcher cost inputs.
DISPATCH_BASE_THOUSAND_RUB_YEAR = 75.0
DISPATCH_RELEASE_THOUSAND_RUB_PER_UNIT = 53.3
DISPATCH_RELEASE_THOUSAND_RUB_PER_UNIT_FORD = .7104


def annual_route_costs(vehicle_type: str, annual_km: float, fleet: int, annual_hours: float) -> dict[str, float]:
    """Return annual cost components in million rubles.

    ``annual_km`` is the model's annual commercial mileage and ``fleet`` is
    the required peak fleet. Infrastructure and dispatching are fixed annual
    route costs from the supplied source table; mileage-dependent items scale
    with annual mileage.
    """
    fuel = annual_km * FUEL_ENERGY_RUB_PER_KM[vehicle_type] / 1e6
    repair = annual_km * REPAIR_RUB_PER_KM[vehicle_type] / 1e6
    crew = annual_hours * DRIVER_HOUR_WITH_CHARGES_RUB[vehicle_type] / 1e6
    return {
        "fuel_energy_mln": fuel,
        "repair_mln": repair,
        "crew_mln": crew,
        "infrastructure_mln": INFRASTRUCTURE_ANNUAL_MLN[vehicle_type],
        "dispatch_mln": DISPATCH_ANNUAL_MLN[vehicle_type],
        "total_before_vehicle": fuel + repair + crew + INFRASTRUCTURE_ANNUAL_MLN[vehicle_type] + DISPATCH_ANNUAL_MLN[vehicle_type],
    }
