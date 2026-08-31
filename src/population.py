"""Download WorldPop 100m population raster and compute population within the city boundary.

WorldPop Global 100m 2020 unconstrained (UN-adjusted) per-country raster.
Download URL pattern:
    https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/PER_COUNTRY/0_Mosaics/RUS/RUS_ppp_v2b_2020_UNadj.tif
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import array_bounds

from config import RAW_DIR, CACHE_DIR, NAMES, WORLDPOP_TIF_PATH


WORLDPOP_URLS = [
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/PER_COUNTRY/0_Mosaics/RUS/RUS_ppp_v2b_2020_UNadj.tif",
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/RUS_plus/RUS_ppp_v2b_2020_UNadj_2.tif",
]

WORLDPOP_TIF_NAME = "RUS_ppp_v2b_2020_UNadj.tif"


def _download_worldpop(urls: list[str], out_path: Path) -> Path:
    import requests
    if out_path.exists():
        print(f"  WorldPop already downloaded: {out_path}")
        return out_path
    for url in urls:
        print(f"  Trying: {url}")
        try:
            r = requests.get(url, stream=True, timeout=60, headers={"User-Agent": "transport-model/0.1"})
            if r.status_code == 200:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                print(f"  Downloaded: {out_path} ({out_path.stat().st_size/1e6:.0f} MB)")
                return out_path
            else:
                print(f"  HTTP {r.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
    raise RuntimeError("Could not download WorldPop raster from any URL")


def compute_population(
    boundary_gdf: gpd.GeoDataFrame,
    raw_dir: Path = RAW_DIR,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
) -> dict:
    out_tif = cache_dir / f"{NAMES['population']}.tif"
    summary_json = cache_dir / f"{NAMES['population']}_summary.json"
    if out_tif.exists() and summary_json.exists() and not force:
        with open(summary_json) as f:
            summary = json.load(f)
        print(f"  Population loaded from cache: {summary['total_population']:.0f}")
        return summary

    tif_path = WORLDPOP_TIF_PATH
    if not tif_path.exists():
        raise FileNotFoundError(f"WorldPop raster not found: {tif_path}")
    print(f"  Using local raster: {tif_path}")

    boundary = boundary_gdf.to_crs("EPSG:4326")
    geom = [boundary.geometry.iloc[0].__geo_interface__]

    print("  Reading and clipping WorldPop raster to boundary ...")
    with rasterio.open(tif_path) as src:
        out_image, out_transform = rio_mask(src, geom, crop=True, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
        )

    out_image = out_image.astype(np.float32)
    valid = out_image[out_image > 0]
    total_pop = float(np.nansum(valid))
    max_pop = float(np.max(valid)) if len(valid) > 0 else 0.0
    mean_pop = float(np.mean(valid)) if len(valid) > 0 else 0.0
    n_cells = int(len(valid))

    cache_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **out_meta) as dst:
        dst.write(out_image)
    print(f"  Saved clipped raster: {out_tif}")

    summary = {
        "total_population": round(total_pop, 1),
        "max_cell_population": round(max_pop, 4),
        "mean_cell_population": round(mean_pop, 4),
        "n_populated_cells": n_cells,
        "raster_path": str(out_tif),
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Population: {total_pop:,.0f} ({n_cells:,} cells)")
    return summary


def load_population_tif() -> np.ndarray:
    tif = CACHE_DIR / f"{NAMES['population']}.tif"
    with rasterio.open(tif) as src:
        return src.read(1)


if __name__ == "__main__":
    from src.boundary import load_boundary

    boundary_gdf = load_boundary(CACHE_DIR / f"{NAMES['boundary']}.geojson")
    summary = compute_population(boundary_gdf)
    print(json.dumps(summary, indent=2))