"""Rolling-stock catalogue and driver-cost parameters for route operation."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True, slots=True)
class VehicleType:
    code: str; name: str; mode: str; capacity_class: str; capacity: float
    unit_cost_mln: float; major_repair_share: float; contract_months: int; service_life_years: int
    annual_contract_cost_mln: float; annual_amortization_mln: float; one_off_cost_mln: float
    technical_readiness: float; electric: bool=False; charging_at_terminal: bool=False
    salary_coefficient: float=1.0; salary_city_coefficient: float=1.0
    work_hours_year: float=1772.0; prep_close_coefficient: float=1.06; ticket_sales_coefficient: float=1.0
    driver_hour_cost: float=0.0; driver_hour_with_charges: float=0.0; annual_crew_cost_mln: float=0.0

_V = [
("ford_transit","Форд-Транзит","Авт","МК",18,2.60,0,.98,256.6,335.7,229),
("gazelle_city","Газель-Сити","Авт","МК",18,3.20,0,.98,256.6,335.7,173),
("paz","ПАЗ","Авт","СК",43,4.10,0,1.05,275.0,359.6,80),
("liaz","ЛИАЗ","Авт","БК",68,12.00,0,1.43,374.5,489.8,73),
("liaz_gas","ЛИАЗ, газовый","Авт","БК",68,12.00,0,1.43,374.5,489.8,73),
("liaz_obk","Лиаз ОБК","Авт","ОБК",93,16.00,0,1.50,392.8,513.8,52),
("liaz_obk_gas","Лиаз ОБК, газовый","Авт","ОБК",93,16.00,0,1.50,392.8,513.8,52),
("kamaz_charge_terminal","Камаз, зарядка на конечной","Элб","БК",72,34.40,.30,1.00,261.9,342.5,50),
("admiral_bk","Адмирал","Тб","БК",73,22.00,0,1.00,261.9,342.5,41),
("admiral_obk","Адмирал ОБК","Тб","ОБК",98,28.00,0,1.10,288.0,376.8,36),
("tuah_bk","БК","ТУАХ","БК",73,27.00,.30,1.00,261.9,342.5,41),
("tuah_obk","Адмирал ОБК","ТУАХ","ОБК",98,33.00,.30,1.10,288.0,376.8,36),
("tm_lvenok","Львенок","Тм","БК",95,45.00,.38,.90,235.7,308.3,30),
("tm_vityaz","Витязь","Тм","ОБК",162,96.00,.38,1.00,261.9,342.5,19),
("tm_2x_bk","2хБК","Тм","2хБК",190,90.00,.38,1.00,261.9,342.5,19),
("tm_3x_bk","3хБК","Тм","3хБК",285,135.00,.38,1.10,288.0,376.8,21),
]
VEHICLE_TYPES = {}
_ANNUAL_CONTRACT = {"ford_transit":.52,"gazelle_city":.64,"paz":.82,"liaz":1.71,"liaz_gas":1.71,"liaz_obk":2.29,"liaz_obk_gas":2.29,"kamaz_charge_terminal":2.98,"admiral_bk":1.47,"admiral_obk":1.87,"tuah_bk":2.34,"tuah_obk":2.86,"tm_lvenok":2.07,"tm_vityaz":4.42,"tm_2x_bk":4.14,"tm_3x_bk":6.21}
_AMORT = {"ford_transit":74,"gazelle_city":69,"paz":39,"liaz":55,"liaz_gas":55,"liaz_obk":50,"liaz_obk_gas":50,"kamaz_charge_terminal":98,"admiral_bk":40,"admiral_obk":41,"tuah_bk":63,"tuah_obk":63,"tm_lvenok":46,"tm_vityaz":57,"tm_2x_bk":54,"tm_3x_bk":81}
_ONE_OFF = {"ford_transit":371.8,"gazelle_city":345.6,"paz":192.7,"liaz":384,"liaz_gas":384,"liaz_obk":352,"liaz_obk_gas":352,"kamaz_charge_terminal":1135,"admiral_bk":594,"admiral_obk":616,"tuah_bk":729,"tuah_obk":726,"tm_lvenok":990,"tm_vityaz":1248,"tm_2x_bk":1170,"tm_3x_bk":1755}
for code,name,mode,cls,cap,cost,repair,scoef,hour,hour_ch,crew in _V:
    VEHICLE_TYPES[code]=VehicleType(code,name,mode,cls,cap,cost,repair,12,15 if mode in ("Тб","ТУАХ") else (30 if mode=="Тм" else 5),_ANNUAL_CONTRACT[code],_AMORT[code],_ONE_OFF[code],.80 if mode=="Авт" else .90,mode in ("Элб","ТУАХ"),code in ("kamaz_charge_terminal","tuah_bk","tuah_obk"),scoef,1.0,1772,1.06,1.0,hour,hour_ch,crew)
DEFAULT_VEHICLE_TYPE="kamaz_charge_terminal"

def get_vehicle_type(code):
    try:return VEHICLE_TYPES[code]
    except KeyError as exc:raise ValueError(f"Unknown vehicle type: {code}") from exc

def round_down_half_minutes(minutes): return math.floor(minutes*2+1e-9)/2
def round_up_to_interval(minutes,interval_min): return math.ceil(minutes/interval_min-1e-12)*interval_min

def calculate_route_operations(*,route_length_km,max_section_flow_pph,vehicle_type=DEFAULT_VEHICLE_TYPE,speed_kmh=18.0,interval_reserve_sec=20.0,terminal_delay_reserve=.08,charging_min_per_terminal=10.0,annual_days=350,park_trip_coefficient=.90,frequency_profile=None):
    """Backward-compatible wrapper around the canonical physical route model."""
    from .route_economics import calculate_route_characteristics
    v=get_vehicle_type(vehicle_type)
    profile=frequency_profile or ((1,.8),(2,1),(7.5,.8),(3,1),(1.5,.8),(3,.5))
    op=calculate_route_characteristics(route_length_km,max_section_flow_pph,capacity_at_4_ppm2=v.capacity,speed_kmh=speed_kmh,interval_reserve_sec=interval_reserve_sec,terminal_delay_reserve=terminal_delay_reserve,charging_min_per_terminal=charging_min_per_terminal,charging_at_terminal=v.charging_at_terminal,technical_readiness=v.technical_readiness,frequency_profile=profile)
    return {"vehicle_type":v.code,"vehicle_name":v.name,"mode":v.mode,"capacity_class":v.capacity_class,"capacity":v.capacity,"unit_cost_mln":v.unit_cost_mln,"major_repair_share":v.major_repair_share,"max_section_flow_pph":float(op.max_section_flow_pph),"frequency_vph":op.frequency_vph,"interval_min":op.interval_min,"cycle_time_min":op.turnaround_min,"release":op.release,"technical_readiness":op.technical_readiness,"fleet":op.fleet,"daily_trips":op.daily_trips,"annual_mileage_km":op.annual_mileage_km,"annual_in_service_hours":op.annual_in_service_hours,"annual_fleet_contract_cost_mln":op.fleet*v.annual_contract_cost_mln,"annual_fleet_amortization_mln":op.fleet*v.annual_amortization_mln,"one_off_fleet_cost_mln":op.fleet*v.one_off_cost_mln}
