"""Operating cost coefficients for the Voronezh TNDP model."""
from __future__ import annotations

FUEL_ENERGY_RUB_PER_KM = {
    "ford_transit":13.45,"gazelle_city":13.45,"paz":28.33,"liaz":42.43,"liaz_gas":22.08,"liaz_obk":49.25,"liaz_obk_gas":25.85,"kamaz_charge_terminal":10.80,"admiral_bk":12.74,"admiral_obk":19.65,"tuah_bk":12.74,"tuah_obk":19.65,"tm_lvenok":14.34,"tm_vityaz":24.43,"tm_2x_bk":28.67,"tm_3x_bk":43.01,
}
REPAIR_RUB_PER_KM = {
    "ford_transit":7.12,"gazelle_city":7.12,"paz":8.58,"liaz":13.18,"liaz_gas":16.51,"liaz_obk":17.71,"liaz_obk_gas":21.07,"kamaz_charge_terminal":20.46,"admiral_bk":20.46,"admiral_obk":25.47,"tuah_bk":20.46,"tuah_obk":25.47,"tm_lvenok":22.69,"tm_vityaz":29.39,"tm_2x_bk":45.83,"tm_3x_bk":68.97,
}

# Infrastructure table: thousand rubles per km/year unless stated otherwise.
DEDICATED_LANE_RUB_KM_YEAR = 1478.0
TRAM_TRACK_RUB_KM_YEAR = 2450.0
CONTACT_NETWORK_RUB_KM_YEAR = 149.55
CONTACT_NETWORK_SHARE = {k:1.0 for k in FUEL_ENERGY_RUB_PER_KM}
CONTACT_NETWORK_SHARE.update({"tuah_bk":0.6,"tuah_obk":0.6})
SUBSTATION_RUB_PER_RELEASE_THOUSAND = 73.59
SUBSTATION_RUB_PER_RELEASE_THOUSAND_OBK = 148.78
CHARGER_RUB_PER_RELEASE_THOUSAND = 650.0

# Source-table totals retained for audit/comparison only.
SOURCE_INFRASTRUCTURE_REFERENCE_MLN = {"ford_transit":30.0,"gazelle_city":30.0,"paz":30.0,"liaz":30.0,"liaz_gas":30.0,"liaz_obk":30.0,"liaz_obk_gas":30.0,"kamaz_charge_terminal":46.0,"admiral_bk":34.0,"admiral_obk":34.0,"tuah_bk":33.0,"tuah_obk":33.0,"tm_lvenok":55.0,"tm_vityaz":54.0,"tm_2x_bk":54.0,"tm_3x_bk":54.0}

DISPATCH_ANNUAL_MLN = {"ford_transit":6.1,"gazelle_city":4.6,"paz":2.0,"liaz":1.3,"liaz_gas":1.3,"liaz_obk":.9,"liaz_obk_gas":.9,"kamaz_charge_terminal":1.4,"admiral_bk":1.1,"admiral_obk":.9,"tuah_bk":1.1,"tuah_obk":.9,"tm_lvenok":.9,"tm_vityaz":.5,"tm_2x_bk":.5,"tm_3x_bk":.5}
DRIVER_HOUR_WITH_CHARGES_RUB = {"ford_transit":335.7,"gazelle_city":335.7,"paz":359.6,"liaz":489.8,"liaz_gas":489.8,"liaz_obk":513.8,"liaz_obk_gas":513.8,"kamaz_charge_terminal":342.5,"admiral_bk":342.5,"admiral_obk":376.8,"tuah_bk":342.5,"tuah_obk":376.8,"tm_lvenok":308.3,"tm_vityaz":342.5,"tm_2x_bk":342.5,"tm_3x_bk":376.8}


def infrastructure_annual_cost_mln(vehicle_type: str, route_length_km: float, fleet: int) -> float:
    """Annual infrastructure cost from route length, fleet and vehicle type."""
    if route_length_km < 0 or fleet < 0:
        raise ValueError("route_length_km and fleet must be non-negative")
    rate = TRAM_TRACK_RUB_KM_YEAR if vehicle_type.startswith("tm_") else DEDICATED_LANE_RUB_KM_YEAR
    total_thousand = route_length_km * rate
    # Contact/cable network and substations are relevant to electric modes.
    if vehicle_type.startswith(("admiral_", "tuah_", "tm_")) or vehicle_type == "kamaz_charge_terminal":
        total_thousand += route_length_km * CONTACT_NETWORK_RUB_KM_YEAR * CONTACT_NETWORK_SHARE[vehicle_type]
        sub_rate = SUBSTATION_RUB_PER_RELEASE_THOUSAND_OBK if ("obk" in vehicle_type or vehicle_type == "kamaz_charge_terminal") else SUBSTATION_RUB_PER_RELEASE_THOUSAND
        total_thousand += fleet * sub_rate
    if vehicle_type == "kamaz_charge_terminal":
        total_thousand += fleet * CHARGER_RUB_PER_RELEASE_THOUSAND
    return total_thousand / 1000.0


def annual_route_costs(vehicle_type: str, annual_km: float, fleet: int, annual_hours: float, route_length_km: float | None = None) -> dict[str, float]:
    """Return annual route operating costs in million rubles."""
    if route_length_km is None:
        raise ValueError("route_length_km is required for infrastructure calculation")
    fuel = annual_km * FUEL_ENERGY_RUB_PER_KM[vehicle_type] / 1e6
    repair = annual_km * REPAIR_RUB_PER_KM[vehicle_type] / 1e6
    crew = annual_hours * DRIVER_HOUR_WITH_CHARGES_RUB[vehicle_type] / 1e6
    infrastructure = infrastructure_annual_cost_mln(vehicle_type, route_length_km, fleet)
    dispatch = DISPATCH_ANNUAL_MLN[vehicle_type]
    return {"fuel_energy_mln":fuel,"repair_mln":repair,"crew_mln":crew,"infrastructure_mln":infrastructure,"dispatch_mln":dispatch,"total_before_vehicle":fuel+repair+crew+infrastructure+dispatch}
