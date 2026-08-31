"""Zone-based TNDP input adapter."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from config import CACHE_DIR, PROJ_EPSG


def load_zone_demand() -> tuple[np.ndarray, gpd.GeoDataFrame]:
    """Load zones and a square OD matrix keyed by stable zone IDs."""
    zones = gpd.read_parquet(CACHE_DIR / "zones" / "zones.parquet").to_crs(PROJ_EPSG).reset_index(drop=True)
    od = pd.read_parquet(CACHE_DIR / "zone_od" / "od_matrix.parquet")
    zone_ids = zones["zone_id"].astype(int).to_numpy()
    aligned = od.reindex(index=zone_ids, columns=zone_ids).fillna(0.0)
    return aligned.to_numpy(dtype=float), zones
