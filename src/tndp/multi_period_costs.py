"""Period-by-period operating costs and annual network aggregation."""
from __future__ import annotations
from dataclasses import asdict
from .interval_profile import DEFAULT_INTERVAL_PROFILE, IntervalPeriod
from .vehicle_types import get_vehicle_type, DRIVER_HOUR_WITH_CHARGES_RUB
from .operating_costs import annual_route_costs
from .route_economics import calculate_route_characteristics


def build_route_period_costs(*, route_length_km: float, peak_flow_pph: float, vehicle_type: str,
                             periods: tuple[IntervalPeriod, ...] = DEFAULT_INTERVAL_PROFILE,
                             speed_kmh: float = 18.0, interval_reserve_sec: float = 20.0,
                             terminal_delay_reserve: float = .08, charging_min_per_terminal: float = 10.0,
                             annual_days: int = 350, park_trip_coefficient: float = .90) -> dict:
    """Build period service and annual cost components for one route."""
    v = get_vehicle_type(vehicle_type)
    out=[]
    for p in periods:
        period_flow = float(peak_flow_pph) * float(p.frequency_factor)
        op = calculate_route_characteristics(
            2.0 * float(route_length_km), period_flow,
            capacity_at_4_ppm2=v.capacity, speed_kmh=speed_kmh,
            interval_reserve_sec=interval_reserve_sec,
            terminal_delay_reserve=terminal_delay_reserve,
            charging_min_per_terminal=charging_min_per_terminal,
            charging_at_terminal=v.charging_at_terminal,
            technical_readiness=v.technical_readiness,
            frequency_profile=((p.hours, 1.0),),
        )
        annual_km = op.annual_mileage_km
        annual_h = op.annual_in_service_hours
        costs = annual_route_costs(vehicle_type, annual_km, op.fleet, annual_h)
        out.append({"number": p.number, "name": p.name, "start": p.start, "end": p.end,
                    "hours": p.hours, "frequency_factor": p.frequency_factor,
                    "peak_flow_pph": peak_flow_pph, "period_flow_pph": period_flow,
                    "frequency_vph": op.frequency_vph, "interval_min": op.interval_min,
                    "turnaround_min": op.turnaround_min, "fleet": op.fleet,
                    "daily_trips": p.hours * op.frequency_vph,
                    "annual_mileage_km": annual_km, "annual_in_service_hours": annual_h,
                    **costs})
    peak_fleet = max((x["fleet"] for x in out), default=0)
    # Fixed vehicle costs are incurred for the peak fleet, not once per period.
    annual_contract = peak_fleet * v.annual_contract_cost_mln
    annual_amort = peak_fleet * v.annual_amortization_mln
    annual_km = sum(x["annual_mileage_km"] for x in out)
    annual_hours = sum(x["annual_in_service_hours"] for x in out)
    annual_fuel = sum(x["fuel_energy_mln"] for x in out)
    annual_repair = sum(x["repair_mln"] for x in out)
    annual_crew = sum(x["crew_mln"] for x in out)
    annual_infra = vcode = out[0]["infrastructure_mln"] if out else 0.0
    annual_dispatch = out[0]["dispatch_mln"] if out else 0.0
    return {"periods": out, "peak_fleet": peak_fleet,
            "annual_mileage_km": annual_km, "annual_in_service_hours": annual_hours,
            "annual_fuel_energy_mln": annual_fuel, "annual_repair_mln": annual_repair,
            "annual_crew_cost_mln": annual_crew, "annual_infrastructure_mln": annual_infra,
            "annual_dispatch_mln": annual_dispatch, "annual_contract_cost_mln": annual_contract,
            "annual_amortization_mln": annual_amort,
            "annual_total_cost_mln": annual_fuel + annual_repair + annual_crew + annual_infra + annual_dispatch + annual_contract + annual_amort,
            "vehicle_type": vehicle_type}
