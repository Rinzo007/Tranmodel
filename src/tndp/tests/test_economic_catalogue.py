from src.tndp.economic_catalog_validation import validate_economic_catalogue
from src.tndp.operating_costs import annual_route_costs
from src.tndp.vehicle_types import VEHICLE_TYPES, calculate_route_operations


def test_all_vehicle_types_have_complete_economic_catalogue():
    report = validate_economic_catalogue()
    assert report["ok"], report
    assert report["vehicle_count"] == 16


def test_all_vehicle_types_produce_positive_operating_plan():
    for code in VEHICLE_TYPES:
        op = calculate_route_operations(
            route_length_km=10.0,
            max_section_flow_pph=400.0,
            vehicle_type=code,
        )
        assert op["fleet"] >= 1
        assert op["annual_mileage_km"] > 0
        assert op["annual_in_service_hours"] > 0
        costs = annual_route_costs(code, op["annual_mileage_km"], op["fleet"], op["annual_in_service_hours"])
        assert costs["total_before_vehicle"] > 0
