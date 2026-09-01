"""Loader for RenatoArbex/TransitNetworkDesign benchmark instances.

The external repository uses nodes, links and demand files plus published route
sets. This module intentionally keeps the format adapter small so benchmark
networks can be used to validate Tranmodel's TNDP solver without depending on
that repository at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_instance(directory: str | Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load <prefix>_nodes.txt, <prefix>_links.txt and <prefix>_demand.txt."""
    root = Path(directory)
    nodes = pd.read_csv(root / f"{prefix}_nodes.txt")
    links = pd.read_csv(root / f"{prefix}_links.txt")
    demand_df = pd.read_csv(root / f"{prefix}_demand.txt")
    n = len(nodes)
    demand = np.zeros((n, n), dtype=float)
    from_col = "from" if "from" in demand_df.columns else demand_df.columns[0]
    to_col = "to" if "to" in demand_df.columns else demand_df.columns[1]
    value_col = "demand" if "demand" in demand_df.columns else demand_df.columns[2]
    for row in demand_df.itertuples(index=False):
        i = int(getattr(row, from_col) if from_col.isidentifier() else row[0]) - 1
        j = int(getattr(row, to_col) if to_col.isidentifier() else row[1]) - 1
        value = float(getattr(row, value_col) if value_col.isidentifier() else row[2])
        if 0 <= i < n and 0 <= j < n:
            demand[i, j] = value
    return nodes, links, demand


def load_route_set(path: str | Path) -> list[tuple[int, ...]]:
    """Parse a literature solution route-set text file.

    The parser accepts the common ``1-2-3-4`` route format and ignores title,
    route-count and frequency lines.
    """
    routes: list[tuple[int, ...]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "-" not in line or not any(ch.isdigit() for ch in line):
            continue
        parts = line.split("-")
        if all(p.strip().isdigit() for p in parts):
            nodes = tuple(int(p.strip()) - 1 for p in parts)
            if len(nodes) >= 2:
                routes.append(nodes)
    return routes
