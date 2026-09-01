from pathlib import Path

ROOT = Path(__file__).resolve().parent

PBF_PATH = Path(r"D:\Programs\Project\central-fed-district-260828.osm.pbf")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
LAYERS_DIR = CACHE_DIR / "layers"
REPORT_DIR = DATA_DIR / "report"

# --- Target city ---
CITY = "Voronezh"
CITY_QUERY = "Воронеж, Воронежская область"
CITY_OSM_RELATION = 1144811
# UTM zone for Voronezh: 37N
PROJ_EPSG = 32637

# --- Reference public transport network ---
# Tracked in the repository so the model does not depend on a developer-local
# D:\... path. The GeoJSON contains the reference stop points, route numbers,
# terminal flags, and route membership used by phase1_real/phase3_real.
REFERENCE_ROUTES_PATH = ROOT / "voronezh_routes_terminals.geojson"

# --- Nominatim ---
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_LOOKUP_URL = "https://nominatim.openstreetmap.org/lookup"
NOMINATIM_USER_AGENT = "transport-model-voronezh/0.1 (dev)"

# --- OSM extraction ---
# Buffer around boundary bbox to keep geometry when clipping roads
OSM_BBOX_MARGIN_DEG = 0.05

# --- WorldPop ---
WORLDPOP_YEAR = 2020
WORLDPOP_COUNTRY = "RUS"
WORLDPOP_TIF_NAME = "rus_pop_2030_CN_100m_R2025A_v1.tif"
WORLDPOP_TIF_PATH = Path(r"D:\Programs\Project\rus_pop_2030_CN_100m_R2025A_v1.tif")

NAMES = {
    "boundary": "boundary_voronezh",
    "roads": "roads",
    "rail": "rail",
    "waterways": "waterways",
    "stops": "stops",
    "pois": "pois",
    "population": "population_voronezh",
}


def ensure_dirs() -> None:
    for d in (RAW_DIR, CACHE_DIR, LAYERS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
