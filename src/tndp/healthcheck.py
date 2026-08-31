"""Lightweight preflight checks for the Tranmodel TNDP pipeline."""
from __future__ import annotations
import importlib
from dataclasses import dataclass

REQUIRED_MODULES = (
    "src.tndp.model", "src.tndp.interval_profile", "src.tndp.period_assignment",
    "src.tndp.multiperiod_assignment", "src.tndp.period_vehicle_plan",
    "src.tndp.period_network_plan", "src.tndp.cost_aggregation",
    "src.tndp.objective", "src.tndp.optimizer",
)

@dataclass(frozen=True, slots=True)
class ImportCheck:
    module: str
    ok: bool
    error: str = ""


def check_imports(modules=REQUIRED_MODULES) -> list[ImportCheck]:
    out=[]
    for name in modules:
        try:
            importlib.import_module(name)
            out.append(ImportCheck(name, True, ""))
        except Exception as exc:
            out.append(ImportCheck(name, False, f"{type(exc).__name__}: {exc}"))
    return out


def assert_imports(modules=REQUIRED_MODULES) -> None:
    failed=[x for x in check_imports(modules) if not x.ok]
    if failed:
        detail="; ".join(f"{x.module}: {x.error}" for x in failed)
        raise RuntimeError(f"Tranmodel import preflight failed: {detail}")
