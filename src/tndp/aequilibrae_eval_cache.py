"""Persistent, bounded cache helpers for expensive AequilibraE evaluations."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any


def stable_route_set_key(route_set, config=None, *, extra: dict[str, Any] | None = None) -> str:
    rows=[]
    for r in getattr(route_set,"routes",[]):
        rows.append({"nodes":list(map(int,r.nodes)),"frequency_vph":float(r.frequency_vph),"vehicle_type":str(r.vehicle_type),"flow":float(r.max_section_flow_pph)})
    payload={"routes":sorted(rows,key=lambda x:(x["nodes"],x["vehicle_type"],x["frequency_vph"])),"config":repr(config),"extra":extra or {}}
    raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"),default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def load_json(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path=Path(cache_dir)/f"{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError,ValueError):
        return None


def save_json(cache_dir: Path, key: str, value: dict[str, Any]) -> Path:
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    tmp=cache_dir/f".{key}.tmp"; path=cache_dir/f"{key}.json"
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); tmp.replace(path)
    return path
