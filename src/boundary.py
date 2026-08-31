"""Download a city boundary polygon from the Nominatim API (cached to disk)."""

import json
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import mapping, shape
from shapely.validation import make_valid

from config import (
    NOMINATIM_LOOKUP_URL,
    NOMINATIM_URL,
    NOMINATIM_USER_AGENT,
)


class BoundaryError(RuntimeError):
    pass


def fetch_boundary_by_relation(
    osm_relation: int,
    out_path: Path,
    force: bool = False,
) -> gpd.GeoDataFrame:
    if out_path.exists() and not force:
        return gpd.read_file(out_path)

    params = {
        "osm_ids": f"R{osm_relation}",
        "format": "json",
        "polygon_geojson": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    r = requests.get(NOMINATIM_LOOKUP_URL, params=params, headers=headers, timeout=60)
    if r.status_code != 200:
        raise BoundaryError(f"Nominatim HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    if not isinstance(data, list) or not data:
        raise BoundaryError(f"Relation R{osm_relation} not found")
    props = data[0]
    geom = props.get("geojson")
    if not geom:
        raise BoundaryError("No polygon geometry in lookup response")
    geom = shape(geom)
    geom = make_valid(geom) if not geom.is_valid else geom

    row = {
        "display_name": props.get("display_name"),
        "osm_type": "relation",
        "osm_id": osm_relation,
        "class": props.get("class"),
        "type": props.get("type"),
        "place_rank": props.get("place_rank"),
    }
    gdf = gpd.GeoDataFrame([row], geometry=[geom], crs="EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return gdf


def _score(feature: dict, osm_relation: int | None) -> int:
    props = feature.get("properties", {})
    score = 0
    if props.get("class") == "boundary" and props.get("type") == "administrative":
        score += 10
    if props.get("osm_type") == "relation":
        score += 3
    if osm_relation is not None:
        if props.get("osm_type") == "relation" and props.get("osm_id") == osm_relation:
            score += 100
    return score


def _extract_geom(feature: dict):
    props = feature.get("properties") or {}
    geom = props.get("geojson") or feature.get("geometry")
    if geom:
        return shape(geom)
    return None


def fetch_boundary(
    query: str,
    out_path: Path,
    osm_relation: int | None = None,
    force: bool = False,
) -> gpd.GeoDataFrame:
    if out_path.exists() and not force:
        return gpd.read_file(out_path)

    params = {
        "q": query,
        "format": "geojson",
        "polygon_geojson": 1,
        "addressdetails": 1,
        "extratags": 1,
        "namedetails": 1,
        "limit": 10,
    }
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=60)
    if r.status_code != 200:
        raise BoundaryError(f"Nominatim HTTP {r.status_code}: {r.text[:300]}")
    features = r.json().get("features", [])
    if not features:
        raise BoundaryError(f"No results for query {query!r}")

    features.sort(key=lambda f: _score(f, osm_relation), reverse=True)
    best = features[0]
    props = best.get("properties", {})
    geom = _extract_geom(best)
    if geom is None or geom.is_empty:
        raise BoundaryError("No polygon geometry returned by Nominatim")
    geom = make_valid(geom) if not geom.is_valid else geom

    row = {
        "display_name": props.get("display_name"),
        "osm_type": props.get("osm_type"),
        "osm_id": props.get("osm_id"),
        "class": props.get("class"),
        "type": props.get("type"),
        "place_rank": props.get("place_rank"),
    }
    gdf = gpd.GeoDataFrame([row], geometry=[geom], crs="EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    time.sleep(1)
    return gdf


def load_boundary(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


if __name__ == "__main__":
    from config import CACHE_DIR, CITY_OSM_RELATION, NAMES

    path = CACHE_DIR / f"{NAMES['boundary']}.geojson"
    gdf = fetch_boundary_by_relation(CITY_OSM_RELATION, path)
    gdf_proj = gdf.to_crs(32637)
    area_km2 = gdf_proj.geometry.area.iloc[0] / 1e6
    main_geom = gdf.geometry.iloc[0]
    print("name:", gdf.display_name.iloc[0])
    print("osm:", gdf.osm_type.iloc[0], gdf.osm_id.iloc[0])
    print(f"area_km2={area_km2:.1f}")
    print(f"geom_type={main_geom.geom_type}, parts={len(main_geom.geoms) if hasattr(main_geom, 'geoms') else 1}")
    print("bounds:", [round(c, 4) for c in main_geom.bounds])