"""Reusable shortest-path cache for the TNDP stop graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import networkx as nx


@dataclass(slots=True)
class StopPathIndex:
    """Shortest paths between the stop nodes actually used by candidates."""

    paths: dict[tuple[int, int], tuple[int, ...]]
    times: dict[tuple[int, int], float]
    lengths_km: dict[tuple[int, int], float]

    def get(self, origin: int, destination: int):
        key = (int(origin), int(destination))
        path = self.paths.get(key)
        if path is None:
            return None
        return path, self.times[key], self.lengths_km[key]


def build_stop_path_index(graph: nx.Graph, stop_mapping, stop_pairs, cache_file: str | Path | None = None) -> StopPathIndex:
    """Build only the OD pairs needed by candidate routes and persist them."""
    pairs = sorted({(int(a), int(b)) for a, b in stop_pairs if int(a) != int(b)})
    if cache_file:
        path = Path(cache_file)
        if path.exists():
            try:
                payload = pickle.loads(path.read_bytes())
                if payload.get("pairs") == pairs:
                    return StopPathIndex(payload["paths"], payload["times"], payload["lengths_km"])
            except Exception:
                pass

    paths: dict[tuple[int, int], tuple[int, ...]] = {}
    times: dict[tuple[int, int], float] = {}
    lengths: dict[tuple[int, int], float] = {}
    for a, b in pairs:
        src, dst = int(stop_mapping[a]), int(stop_mapping[b])
        try:
            p = tuple(nx.shortest_path(graph, src, dst, weight="time"))
        except nx.NetworkXNoPath:
            continue
        paths[(a, b)] = p
        times[(a, b)] = float(nx.path_weight(graph, p, weight="time"))
        lengths[(a, b)] = float(nx.path_weight(graph, p, weight="length_km"))

    index = StopPathIndex(paths, times, lengths)
    if cache_file:
        path = Path(cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"pairs": pairs, "paths": paths, "times": times, "lengths_km": lengths}, protocol=pickle.HIGHEST_PROTOCOL))
    return index
