from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from railing_removal.atomic_orientation_writeback import (
    PROTECTED_CLEANED_PLY,
)


PRODUCTION_BLEND_FILENAME = (
    "plant-cleaned-garden-ec2fbd1-final-v2-orientation-review-v1.blend"
)


def _read_projects(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.resolve().read_text(encoding="utf-8-sig"))
    if value.get("schema_version") != 1:
        raise ValueError("project manifest schema_version must be 1")
    projects = value.get("projects")
    if not isinstance(projects, list) or not projects:
        raise ValueError("project manifest requires projects")
    return projects


def build_production_orientation_manifest(
    projects_manifest: Path,
    production_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Plan versioned Blender reviews without opening any source for writing."""

    production_root = production_root.resolve()
    output = output.resolve()
    if not production_root.is_dir():
        raise NotADirectoryError(production_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output}")

    scans: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for item in _read_projects(projects_manifest):
        scan_id = str(item.get("scan_id", "")).strip()
        project = Path(item["project"]).resolve()
        scan_dir = project.parent
        if not scan_id or scan_dir.name != scan_id:
            raise ValueError("project scan ID must match its directory")
        if scan_dir.parent != production_root:
            raise ValueError(f"project is outside production root: {project}")
        source = scan_dir / PROTECTED_CLEANED_PLY
        if source.is_symlink():
            raise ValueError(f"refusing linked cleaned PLY: {source}")
        if not source.is_file():
            missing.append(
                {
                    "scan_id": scan_id,
                    "reason": "missing_cleaned_ply",
                    "expected_source": str(source.resolve()),
                }
            )
            continue
        blend_output = scan_dir / PRODUCTION_BLEND_FILENAME
        if blend_output.exists():
            raise FileExistsError(
                f"existing production Blender review: {blend_output}"
            )
        scans.append(
            {
                "scan_id": scan_id,
                "source": str(source.resolve()),
                "source_opened_read_only": True,
                "blend_output": str(blend_output.resolve()),
            }
        )

    manifest = {
        "schema_version": 1,
        "projects_manifest": str(projects_manifest.resolve()),
        "production_root": str(production_root),
        "source_files_opened_read_only": True,
        "blend_filename": PRODUCTION_BLEND_FILENAME,
        "summary": {
            "eligible": len(scans),
            "missing_cleaned_ply": len(missing),
        },
        "scans": scans,
        "missing": missing,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return manifest
