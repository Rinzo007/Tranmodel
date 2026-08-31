from __future__ import annotations

from pathlib import Path


def ensure_transit_project(project_dir: str | Path, progress=None) -> Path:
    """Create a minimal AequilibraE project for transit assignment.

    The full OSM road graph is intentionally not imported here. Tranmodel
    handles route construction on its own cached graph; AequilibraE is used
    only for transit graph/assignment evaluation.
    """
    notify = progress or (lambda _: None)
    project_dir = Path(project_dir)
    marker = project_dir / "tranmodel_transit_only.txt"
    if project_dir.exists() and marker.exists():
        notify("Минимальный Transit-проект найден в кэше.")
        return project_dir
    if project_dir.exists():
        import shutil
        shutil.rmtree(project_dir)
    try:
        from aequilibrae import Project
    except ImportError as exc:
        raise RuntimeError("AequilibraE не установлен") from exc
    project = Project()
    project.new(project_dir)
    project.close()
    marker.write_text("transit-only\n", encoding="utf-8")
    notify("Минимальный Transit-проект создан.")
    return project_dir
