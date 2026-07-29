from __future__ import annotations

import json
from pathlib import Path

from railing_removal.camera_inventory_batch import (
    run_camera_inventory_batch,
)


def _manifest(path: Path, projects: list[Path]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {
                        "scan_id": f"scan-{index}",
                        "project": str(project),
                    }
                    for index, project in enumerate(projects, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_camera_inventory_batch_is_read_only_and_resumable(
    tmp_path: Path,
) -> None:
    projects = [tmp_path / "one.psx", tmp_path / "two.psx"]
    for project in projects:
        project.write_text("immutable source", encoding="utf-8")
    manifest = tmp_path / "projects.json"
    _manifest(manifest, projects)
    output = tmp_path / "inventories"
    existing = output / "scan-1"
    existing.mkdir(parents=True)
    (existing / "camera-inventory.json").write_text(
        json.dumps(
            {
                "project_opened_read_only": True,
                "project": str(projects[0]),
            }
        ),
        encoding="utf-8",
    )
    calls: list[Path] = []

    def inventory(project: Path) -> dict[str, object]:
        calls.append(project)
        return {
            "schema_version": 1,
            "project": str(project),
            "project_opened_read_only": True,
            "camera_count": 12,
        }

    report = run_camera_inventory_batch(
        manifest,
        output,
        inventory_project=inventory,
    )

    assert calls == [projects[1].resolve()]
    assert report["summary"] == {
        "complete": 2,
        "partial": 0,
        "failed": 0,
    }
    assert json.loads(
        (output / "scan-2" / "camera-inventory.json").read_text(
            encoding="utf-8"
        )
    )["project_opened_read_only"]
    assert [project.read_text(encoding="utf-8") for project in projects] == [
        "immutable source",
        "immutable source",
    ]


def test_camera_inventory_batch_preserves_partial_directory(
    tmp_path: Path,
) -> None:
    project = tmp_path / "one.psx"
    project.write_text("immutable source", encoding="utf-8")
    manifest = tmp_path / "projects.json"
    _manifest(manifest, [project])
    output = tmp_path / "inventories"
    partial = output / "scan-1"
    partial.mkdir(parents=True)
    evidence = partial / "evidence.txt"
    evidence.write_text("keep me", encoding="utf-8")

    def forbidden(project: Path) -> dict[str, object]:
        raise AssertionError(f"must not process partial {project}")

    report = run_camera_inventory_batch(
        manifest,
        output,
        inventory_project=forbidden,
    )

    assert report["summary"] == {
        "complete": 0,
        "partial": 1,
        "failed": 0,
    }
    assert evidence.read_text(encoding="utf-8") == "keep me"


def test_camera_inventory_batch_accepts_windows_utf8_bom(
    tmp_path: Path,
) -> None:
    project = tmp_path / "one.psx"
    project.write_text("immutable source", encoding="utf-8")
    manifest = tmp_path / "projects.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {"scan_id": "scan-1", "project": str(project)}
                ],
            }
        ),
        encoding="utf-8-sig",
    )

    report = run_camera_inventory_batch(
        manifest,
        tmp_path / "output",
        inventory_project=lambda path: {
            "project": str(path),
            "project_opened_read_only": True,
        },
    )

    assert report["summary"]["complete"] == 1
