"""Unified AequilibraE workflow including public transport assignment."""

from __future__ import annotations

import json

from src.aequilibrae_pipeline import PROJECT_DIR, _open_project, run_all
from src.aequilibrae_transit import TransitPipelineError, run_transit_assignment


def run_full_model(force: bool = False) -> dict:
    """Run road distribution/assignment and then assign the same OD demand to PT."""
    report = run_all(force=force)

    project = _open_project()
    try:
        demand = project.matrices.get_matrix("demand_gravity")
        transit_report = run_transit_assignment(project, demand, force=force)
    finally:
        project.close()

    report["transit"] = transit_report
    report["project"] = str(PROJECT_DIR)
    path = PROJECT_DIR.parent / "aequilibrae_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


__all__ = ["run_full_model", "TransitPipelineError"]
