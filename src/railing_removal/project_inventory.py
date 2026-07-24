from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_project_manifest(
    source_root: Path,
    output: Path,
    *,
    photo_root: Path | None = None,
) -> dict[str, Any]:
    """Index only projects physically beneath one explicit source root."""

    source_root = source_root.resolve()
    output = output.resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output}")
    if photo_root is not None:
        photo_root = photo_root.resolve()
        if not photo_root.is_dir():
            raise NotADirectoryError(photo_root)

    projects: list[dict[str, Any]] = []
    scan_ids: set[str] = set()
    for project in sorted(source_root.rglob("*.psx")):
        scan_id = project.parent.name
        if scan_id in scan_ids:
            raise ValueError(f"duplicate scan directory name: {scan_id}")
        scan_ids.add(scan_id)
        photos = photo_root / scan_id if photo_root is not None else None
        projects.append(
            {
                "scan_id": scan_id,
                "project": str(project.resolve()),
                "photo_dir": (
                    str(photos.resolve())
                    if photos is not None and photos.is_dir()
                    else None
                ),
            }
        )
    if not projects:
        raise ValueError(f"no Metashape projects found under {source_root}")

    manifest = {
        "schema_version": 1,
        "source_root": str(source_root),
        "photo_root": str(photo_root) if photo_root is not None else None,
        "project_count": len(projects),
        "projects": projects,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return manifest
