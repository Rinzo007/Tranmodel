from src.tndp.aequilibrae_eval_cache import load_json, save_json, stable_route_set_key
from src.tndp.model import Route, RouteSet


def test_route_set_key_is_stable_and_changes_on_frequency():
    a=RouteSet([Route((1,2,3),frequency_vph=6.0,vehicle_type="liaz")])
    b=RouteSet([Route((1,2,3),frequency_vph=6.0,vehicle_type="liaz")])
    c=RouteSet([Route((1,2,3),frequency_vph=7.0,vehicle_type="liaz")])
    assert stable_route_set_key(a)==stable_route_set_key(b)
    assert stable_route_set_key(a)!=stable_route_set_key(c)


def test_json_cache_roundtrip(tmp_path):
    key="abc123"
    value={"score":12.5,"periods":[1,2,3]}
    save_json(tmp_path,key,value)
    assert load_json(tmp_path,key)==value
