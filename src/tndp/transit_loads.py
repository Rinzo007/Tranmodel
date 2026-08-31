from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def extract_transit_segment_loads(project_path: str | Path, transit_results: Any) -> dict[str, Any]:
    """Map AequilibraE TransitAssignmentResults link loads to GTFS route segments.

    AequilibraE's TransitAssignmentResults exposes graph link loads through
    ``get_load_results()``.  The public-transport database contains the
    route_links table (route/pattern/sequence) and the routes table.  We join
    those tables to the assignment link ids and return route-level segment
    loads.  No OD reconstruction is involved here.
    """
    db = Path(project_path) / "public_transport.sqlite"
    if not db.exists():
        raise FileNotFoundError(db)

    load_obj = transit_results.get_load_results()
    if isinstance(load_obj, pd.DataFrame):
        loads = load_obj.copy()
    elif hasattr(load_obj, "data"):
        data = load_obj.data
        try:
            loads = pd.DataFrame(data)
        except Exception:
            loads = pd.DataFrame({"volume": np.asarray(data["volume"])})
        if hasattr(load_obj, "index"):
            try:
                loads.index = np.asarray(load_obj.index)
            except Exception:
                pass
    else:
        raise TypeError("Unsupported AequilibraE transit load result type")

    if "volume" not in loads.columns:
        candidates = [c for c in loads.columns if str(c).lower() in {"flow", "flows", "load", "total_flow"}]
        if not candidates:
            raise ValueError(f"Transit load result has no volume column: {list(loads.columns)}")
        loads = loads.rename(columns={candidates[0]: "volume"})

    link_ids = np.asarray(loads.index)
    volumes = pd.to_numeric(loads["volume"], errors="coerce").fillna(0.0).to_numpy(float)
    load_df = pd.DataFrame({"assignment_link_id": link_ids, "volume": volumes})

    with sqlite3.connect(db) as con:
        route_links = pd.read_sql_query(
            "SELECT transit_link, pattern_id, seq, from_stop, to_stop, distance FROM route_links",
            con,
        )
        routes = pd.read_sql_query(
            "SELECT pattern_id, route_id FROM routes",
            con,
        )

    if route_links.empty:
        raise ValueError("AequilibraE route_links table is empty")

    route_links["transit_link"] = pd.to_numeric(route_links["transit_link"], errors="coerce")
    route_links["pattern_id"] = pd.to_numeric(route_links["pattern_id"], errors="coerce")
    route_links["seq"] = pd.to_numeric(route_links["seq"], errors="coerce")
    routes["pattern_id"] = pd.to_numeric(routes["pattern_id"], errors="coerce")

    merged = route_links.merge(routes, on="pattern_id", how="left")
    merged = merged.merge(load_df, left_on="transit_link", right_on="assignment_link_id", how="left")
    merged["volume"] = merged["volume"].fillna(0.0)

    matched = int(merged["assignment_link_id"].notna().sum())
    if matched == 0:
        # Some versions expose graph link ids as pattern_mapping links.
        with sqlite3.connect(db) as con:
            mapping = pd.read_sql_query(
                "SELECT pattern_id, seq, link, dir FROM pattern_mapping", con
            )
        mapping["link"] = pd.to_numeric(mapping["link"], errors="coerce")
        fallback = mapping.merge(load_df, left_on="link", right_on="assignment_link_id", how="left")
        fallback["volume"] = fallback["volume"].fillna(0.0)
        fallback = fallback.merge(routes, on="pattern_id", how="left")
        if fallback["assignment_link_id"].notna().any():
            merged = merged.drop(columns=["volume", "assignment_link_id"])
            merged = merged.merge(
                fallback[["pattern_id", "seq", "route_id", "volume"]],
                on=["pattern_id", "seq", "route_id"], how="left", suffixes=("", "_fallback")
            )
            merged["volume"] = merged["volume_fallback"].fillna(merged["volume"]).fillna(0.0)
            matched = int(fallback["assignment_link_id"].notna().sum())

    merged["route_index"] = merged["route_id"].astype(str).str.extract(r"(\d+)", expand=False)
    merged["route_index"] = pd.to_numeric(merged["route_index"], errors="coerce")
    merged["segment_index"] = merged["seq"].astype(int) - 1

    route_segments: dict[int, list[dict[str, Any]]] = {}
    for row in merged.dropna(subset=["route_index"]).itertuples(index=False):
        ri = int(row.route_index)
        route_segments.setdefault(ri, []).append({
            "segment_index": int(row.segment_index),
            "from_stop": int(row.from_stop),
            "to_stop": int(row.to_stop),
            "volume_pph": float(row.volume),
            "assignment_link_id": int(row.transit_link),
            "distance_m": float(row.distance),
        })

    max_sections: dict[int, dict[str, Any]] = {}
    for ri, segments in route_segments.items():
        if not segments:
            continue
        best = max(segments, key=lambda x: x["volume_pph"])
        max_sections[ri] = {
            "max_section_flow_pph": best["volume_pph"],
            "max_section_index": best["segment_index"],
            "from_stop": best["from_stop"],
            "to_stop": best["to_stop"],
        }

    return {
        "source": "AequilibraE TransitAssignmentResults",
        "matched_assignment_links": matched,
        "assignment_link_count": int(len(load_df)),
        "route_link_count": int(len(route_links)),
        "route_segments": route_segments,
        "max_sections": max_sections,
    }
