from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _first_existing(columns, names):
    lower = {str(c).lower(): c for c in columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _load_dataframe(result: Any) -> pd.DataFrame:
    """Normalize TransitAssignmentResults.get_load_results() across AequilibraE versions."""
    raw = result.get_load_results()
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    elif hasattr(raw, "data"):
        try:
            df = pd.DataFrame(raw.data)
        except Exception as exc:
            raise TypeError("Cannot convert AequilibraE transit load result to DataFrame") from exc
        if hasattr(raw, "index"):
            try:
                df.index = np.asarray(raw.index)
            except Exception:
                pass
    else:
        raise TypeError(f"Unsupported transit load result type: {type(raw)!r}")

    link_col = _first_existing(df.columns, ["link_id", "link", "transit_link"])
    if link_col is not None:
        df["assignment_link_id"] = pd.to_numeric(df[link_col], errors="coerce")
    else:
        df["assignment_link_id"] = pd.to_numeric(pd.Index(df.index), errors="coerce")

    volume_col = _first_existing(df.columns, ["volume", "demand_tot", "total_flow", "flow", "load"])
    if volume_col is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        numeric = [c for c in numeric if str(c).lower() not in {"link_id", "link", "transit_link", "assignment_link_id"}]
        if not numeric:
            raise ValueError(f"Cannot find transit load column in {list(df.columns)}")
        volume_col = numeric[0]
    df["volume"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)
    return df[["assignment_link_id", "volume"]].dropna(subset=["assignment_link_id"])


def _read_route_model(db: Path):
    with sqlite3.connect(db) as con:
        route_links = pd.read_sql_query("SELECT * FROM route_links", con)
        routes = pd.read_sql_query("SELECT * FROM routes", con)
        mapping = pd.read_sql_query("SELECT * FROM pattern_mapping", con) if _table_exists(con, "pattern_mapping") else pd.DataFrame()
    return route_links, routes, mapping


def _table_exists(con, table: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _build_links(route_links: pd.DataFrame, routes: pd.DataFrame, mapping: pd.DataFrame, loads: pd.DataFrame) -> pd.DataFrame:
    if route_links.empty:
        raise ValueError("AequilibraE route_links table is empty")

    # AequilibraE's route_links is the authoritative route/pattern/sequence
    # model. We deliberately do not infer route identity from route_id text.
    for col in ["transit_link", "pattern_id", "seq"]:
        if col in route_links:
            route_links[col] = pd.to_numeric(route_links[col], errors="coerce")
    if "pattern_id" in routes:
        routes["pattern_id"] = pd.to_numeric(routes["pattern_id"], errors="coerce")

    merged = route_links.copy()
    if "pattern_id" in merged.columns and "pattern_id" in routes.columns:
        route_name_cols = [c for c in ["pattern_id", "route_id", "route_short_name", "route_long_name"] if c in routes.columns]
        merged = merged.merge(routes[route_name_cols].drop_duplicates("pattern_id"), on="pattern_id", how="left")

    merged = merged.merge(loads, left_on="transit_link", right_on="assignment_link_id", how="left")
    matched = int(merged["assignment_link_id"].notna().sum())

    # In some project versions the assignment link is represented by the
    # pattern_mapping link rather than route_links.transit_link.
    if matched == 0 and not mapping.empty and {"pattern_id", "seq", "link"}.issubset(mapping.columns):
        mapping = mapping.copy()
        for col in ["pattern_id", "seq", "link"]:
            mapping[col] = pd.to_numeric(mapping[col], errors="coerce")
        fb = mapping.merge(loads, left_on="link", right_on="assignment_link_id", how="inner")
        if not fb.empty:
            fb = fb[["pattern_id", "seq", "volume", "assignment_link_id"]].drop_duplicates(["pattern_id", "seq"])
            merged = merged.drop(columns=["volume", "assignment_link_id"], errors="ignore")
            merged = merged.merge(fb, on=["pattern_id", "seq"], how="left")
            matched = int(merged["assignment_link_id"].notna().sum())

    merged["volume"] = pd.to_numeric(merged.get("volume", 0.0), errors="coerce").fillna(0.0)
    return merged, matched


def extract_transit_segment_loads(project_path: str | Path, transit_results: Any) -> dict[str, Any]:
    """Return exact AequilibraE transit loads grouped by route and segment.

    The function intentionally uses ``TransitAssignmentResults.get_load_results``
    rather than reconstructing flows from OD demand. AequilibraE documents this
    method as translating graph assignment results into network-level link loads.
    """
    db = Path(project_path) / "public_transport.sqlite"
    if not db.exists():
        raise FileNotFoundError(db)

    loads = _load_dataframe(transit_results)
    route_links, routes, mapping = _read_route_model(db)
    merged, matched = _build_links(route_links, routes, mapping, loads)

    if "seq" not in merged.columns:
        raise ValueError("route_links has no seq field")
    merged["seq"] = pd.to_numeric(merged["seq"], errors="coerce")
    merged["segment_index"] = merged["seq"].fillna(1).astype(int) - 1

    # Stable route identity: pattern_id is retained even when route_id is not
    # numeric, so the evaluator can match the generated GTFS pattern reliably.
    merged["route_key"] = merged.get("pattern_id", pd.Series(index=merged.index, dtype=float)).astype("Int64").astype(str)
    if "route_id" in merged.columns:
        merged["route_id"] = merged["route_id"].astype(str)

    route_segments: dict[str, list[dict[str, Any]]] = {}
    for row in merged.itertuples(index=False):
        key = str(getattr(row, "route_key"))
        if key in {"<NA>", "nan", "None"}:
            continue
        segment = {
            "segment_index": int(getattr(row, "segment_index")),
            "volume_pph": float(getattr(row, "volume")),
            "assignment_link_id": int(getattr(row, "assignment_link_id")) if pd.notna(getattr(row, "assignment_link_id")) else None,
        }
        for field in ("from_stop", "to_stop", "distance", "route_id", "route_short_name", "route_long_name", "pattern_id"):
            if hasattr(row, field):
                value = getattr(row, field)
                if pd.notna(value):
                    segment[field] = value.item() if hasattr(value, "item") else value
        if "distance" in segment:
            segment["distance_m"] = float(segment.pop("distance"))
        route_segments.setdefault(key, []).append(segment)

    max_sections = {}
    for key, segments in route_segments.items():
        if not segments:
            continue
        best = max(segments, key=lambda x: x["volume_pph"])
        max_sections[key] = {
            "max_section_flow_pph": best["volume_pph"],
            "max_section_index": best["segment_index"],
            "from_stop": best.get("from_stop"),
            "to_stop": best.get("to_stop"),
            "pattern_id": best.get("pattern_id"),
            "route_id": best.get("route_id"),
        }

    return {
        "source": "AequilibraE TransitAssignmentResults.get_load_results",
        "matched_assignment_links": matched,
        "assignment_link_count": int(len(loads)),
        "route_link_count": int(len(route_links)),
        "match_rate": float(matched / max(len(route_links), 1)),
        "route_segments": route_segments,
        "max_sections": max_sections,
    }
