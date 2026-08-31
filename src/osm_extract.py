"""Extract road, rail, waterway, stop and POI layers from an OSM PBF file
and clip them to the city boundary.

Two-pass pyosmium reading:
  pass 1 (node blocks only): keep coordinates of every node inside an enlarged
      bounding box, plus stand-alone stop/POI points;
  pass 2 (way blocks only): assemble geometries of candidate ways from the
      stored coords and keep those intersecting the bounding box.

Memory stays proportional to the number of nodes inside the extended bbox.
Intermediate results are cached as pickle so repeated runs skip PBF reads.
"""

import pickle
import time
from pathlib import Path

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, box

from config import LAYERS_DIR, OSM_BBOX_MARGIN_DEG, CACHE_DIR

# ---------------------------------------------------------------------------
# Tag profiles
# ---------------------------------------------------------------------------
ROAD_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "service", "road", "track", "pedestrian", "footway",
    "cycleway", "services",
}

RAILWAYS = {
    "rail", "tram", "light_rail", "subway", "monorail", "narrow_gauge",
    "funicular",
}

WATERWAYS = {"river", "canal", "stream"}

POI_KEYS = {
    "amenity", "shop", "tourism", "leisure", "craft", "office",
    "healthcare", "education", "aeroway",
}

ROW_KEYS = {
    "roads": ["highway", "name", "oneway", "maxspeed", "access"],
    "rail": ["railway", "name", "service"],
    "waterways": ["waterway", "name"],
    "stops": ["kind", "name", "public_transport", "railway", "highway", "amenity"],
}


def _node_kind(tags: dict) -> str | None:
    t = tags.get
    if t("railway") in {"station", "halt", "tram_stop"}:
        return "rail"
    pt = t("public_transport")
    if pt in ("station", "stop_position", "platform"):
        if t("railway"):
            return "rail"
        if pt == "stop_position" and (t("busway") or t("highway") == "bus_stop"):
            return "bus"
        return "platform"
    if t("highway") == "bus_stop":
        return "bus"
    if t("amenity") in {"bus_station", "ferry_terminal"}:
        return "ferry" if t("amenity") == "ferry_terminal" else "bus"
    return None


def _way_category(tags: dict) -> str | None:
    t = tags.get
    if t("highway") in ROAD_HIGHWAYS and t("area") != "yes":
        return "roads"
    if t("railway") in RAILWAYS:
        return "rail"
    if t("waterway") in WATERWAYS:
        return "waterways"
    if t("public_transport") in {"platform", "station"} or t("railway") in {"platform"}:
        return "stops"
    return None


class _NodePass(osmium.SimpleHandler):
    """Pass 1: coords + stop/POI points inside the extended bbox."""

    def __init__(self, bounds: tuple[float, float, float, float]):
        super().__init__()
        self.minx, self.miny, self.maxx, self.maxy = bounds
        self.coords: dict[int, tuple[float, float]] = {}
        self.stop_points: list[dict] = []
        self.poi_points: list[dict] = []
        self.n = 0

    def node(self, obj):
        self.n += 1
        if self.n % 5_000_000 == 0:
            print(f"  nodes: {self.n/1e6:.1f}M, coords: {len(self.coords)/1e6:.1f}M ...")
        lon, lat = obj.location.lon, obj.location.lat
        if not (self.minx <= lon <= self.maxx and self.miny <= lat <= self.maxy):
            return
        tags = dict(obj.tags)
        self.coords[obj.id] = (lon, lat)
        kind = _node_kind(tags)
        if kind is not None:
            self.stop_points.append({
                "osm_id": obj.id, "kind": kind,
                "name": tags.get("name"),
                "public_transport": tags.get("public_transport"),
                "railway": tags.get("railway"),
                "highway": tags.get("highway"),
                "amenity": tags.get("amenity"),
                "geometry": Point(lon, lat),
            })
        elif tags and any(k in tags for k in POI_KEYS):
            main_key = next(k for k in POI_KEYS if k in tags)
            self.poi_points.append({
                "osm_id": obj.id, "name": tags.get("name"),
                "main_key": main_key, "value": tags.get(main_key),
                "geometry": Point(lon, lat),
            })


class _WayPass(osmium.SimpleHandler):
    """Pass 2: geometries of candidate ways intersecting the bbox."""

    def __init__(self, coords: dict, bbox_poly: Polygon):
        super().__init__()
        self.coords = coords
        self.bbox_poly = bbox_poly
        self.rows: dict[str, list] = {c: [] for c in ROW_KEYS}
        self.n = 0

    def way(self, obj):
        self.n += 1
        if self.n % 2_000_000 == 0:
            print(f"  ways: {self.n/1e6:.1f}M ...")
        tags = dict(obj.tags)
        cat = _way_category(tags)
        if cat is None:
            return
        pts = [self.coords[r.ref] for r in obj.nodes if r.ref in self.coords]
        if len(pts) < 2:
            return
        geom = LineString(pts)
        if not geom.is_valid or not geom.intersects(self.bbox_poly):
            return
        row = {"osm_id": obj.id}
        for k in ROW_KEYS[cat]:
            if k == "kind":
                row[k] = _node_kind(tags) or "platform"
            else:
                row[k] = tags.get(k)
        row["geometry"] = geom
        self.rows[cat].append(row)


def _intermediate_cache_dir() -> Path:
    p = CACHE_DIR / "intermediate"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_intermediate(data: dict, name: str) -> None:
    p = _intermediate_cache_dir() / f"{name}.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)
    print(f"  cached {name} -> {p} ({p.stat().st_size/1e6:.1f} MB)")


def _load_intermediate(name: str):
    p = _intermediate_cache_dir() / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def extract_layers(
    pbf_path: Path,
    boundary: Polygon,
    out_dir: Path,
    force: bool = False,
) -> dict[str, gpd.GeoDataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ("roads", "rail", "waterways", "stops", "pois")
    cache_files = {n: out_dir / f"{n}.parquet" for n in names}
    if all(f.exists() for f in cache_files.values()) and not force:
        return {n: gpd.read_parquet(out_dir / f"{n}.parquet") for n in names}

    minx, miny, maxx, maxy = boundary.bounds
    bounds = (
        minx - OSM_BBOX_MARGIN_DEG, miny - OSM_BBOX_MARGIN_DEG,
        maxx + OSM_BBOX_MARGIN_DEG, maxy + OSM_BBOX_MARGIN_DEG,
    )
    bbox_poly = box(*bounds)

    # --- Pass 1: nodes ---
    npass_data = _load_intermediate("pass1_nodes")
    if npass_data is None:
        print("== pass 1/2 (nodes in bbox) ==")
        t0 = time.time()
        npass = _NodePass(bounds)
        npass.apply_file(str(pbf_path))
        print(f"  done {time.time()-t0:.0f}s | stop points={len(npass.stop_points)}, "
              f"poi points={len(npass.poi_points)}, coords={len(npass.coords)}")
        npass_data = {
            "coords": npass.coords,
            "stop_points": npass.stop_points,
            "poi_points": npass.poi_points,
        }
        _save_intermediate(npass_data, "pass1_nodes")
        del npass
    else:
        print("== pass 1/2: loaded from cache ==")

    # --- Pass 2: ways ---
    wpass_data = _load_intermediate("pass2_ways")
    if wpass_data is None:
        print("== pass 2/2 (ways in bbox) ==")
        t0 = time.time()
        wpass = _WayPass(npass_data["coords"], bbox_poly)
        wpass.apply_file(str(pbf_path))
        print(f"  done {time.time()-t0:.0f}s | ways: " +
              ", ".join(f"{k}={len(v)}" for k, v in wpass.rows.items()))
        wpass_data = dict(wpass.rows)
        _save_intermediate(wpass_data, "pass2_ways")
        del wpass
    else:
        print("== pass 2/2: loaded from cache ==")

    del npass_data["coords"]

    # --- Assemble GeoDataFrames ---
    gdfs: dict[str, gpd.GeoDataFrame] = {}

    for cat in ("roads", "rail", "waterways"):
        rows = wpass_data.get(cat, [])
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        if not gdf.empty:
            gdf = gpd.clip(gdf, boundary, keep_geom_type=True)
        gdfs[cat] = gdf

    # --- Stops: ways + point nodes ---
    stop_rows = wpass_data.get("stops", [])
    gdfs["stops"] = _build_points_gdf(stop_rows, npass_data["stop_points"], boundary)

    # --- POIs: point nodes only ---
    gdfs["pois"] = _build_pois_gdf(npass_data["poi_points"], boundary)

    for n in names:
        gdfs[n].to_parquet(out_dir / f"{n}.parquet")
        print(f"  saved {n}: {len(gdfs[n])} features")
    return gdfs


def _build_points_gdf(way_rows: list, point_rows: list, boundary: Polygon) -> gpd.GeoDataFrame:
    all_rows: list[dict] = []
    for r in way_rows:
        all_rows.append(r)
    for pr in point_rows:
        all_rows.append(pr)
    if not all_rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(all_rows, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf.geometry.within(boundary)].reset_index(drop=True)
    gdf["kind"] = gdf["kind"].fillna("platform")
    return gdf


def _build_pois_gdf(point_rows: list, boundary: Polygon) -> gpd.GeoDataFrame:
    if not point_rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(point_rows, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf.geometry.within(boundary)].reset_index(drop=True)
    return gdf


if __name__ == "__main__":
    from src.boundary import load_boundary
    from config import CACHE_DIR, LAYERS_DIR, NAMES, PBF_PATH

    boundary = load_boundary(CACHE_DIR / f"{NAMES['boundary']}.geojson").geometry.iloc[0]
    layers = extract_layers(PBF_PATH, boundary, LAYERS_DIR)
    print("EXTRACT OK")
    for n, gdf in layers.items():
        print(f"  {n}: {len(gdf)}")