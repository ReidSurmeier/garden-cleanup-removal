from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from railing_removal.publish_outputs import publish_outputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publish_outputs_only_adds_versioned_derived_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "scans"
    scan_root = source_root / "scan-a"
    scan_root.mkdir(parents=True)
    project = scan_root / "scan-a.psx"
    project.write_bytes(b"immutable project")
    source_photo = scan_root / "source.jpg"
    source_photo.write_bytes(b"immutable photo")
    project_hash = _sha256(project)
    photo_hash = _sha256(source_photo)
    projects = tmp_path / "projects.json"
    projects.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_root": str(source_root),
                "projects": [
                    {"scan_id": "scan-a", "project": str(project)}
                ],
            }
        ),
        encoding="utf-8",
    )
    cleanup = tmp_path / "cleanup" / "scan-a"
    final = cleanup / "final"
    (final / "render-source").mkdir(parents=True)
    (final / "render-plant").mkdir(parents=True)
    (final / "plant-cleaned-color-corrected.ply").write_bytes(b"clean")
    (final / "render-source" / "front-rgb.png").write_bytes(b"before")
    (final / "render-plant" / "front-rgb.png").write_bytes(b"after")
    (cleanup / "run-report.json").write_text("{}\n", encoding="utf-8")
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "scan_id": "scan-a",
                        "status": "complete",
                        "output": str(cleanup),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "publish.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_tag": "garden-abc123-stride8",
                "scans": [
                    {"scan_id": "scan-a", "action": "publish_clean"}
                ],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "publish-report.json"

    report = publish_outputs(projects, batch, plan, report_path)

    assert report["summary"] == {"publish_clean": 1, "publish_flag": 0}
    assert (
        scan_root / "plant-cleaned-garden-abc123-stride8.ply"
    ).read_bytes() == b"clean"
    assert (
        scan_root / "plant-cleanup-before-garden-abc123-stride8.png"
    ).read_bytes() == b"before"
    assert _sha256(project) == project_hash
    assert _sha256(source_photo) == photo_hash

    with pytest.raises(FileExistsError, match="overwrite"):
        publish_outputs(projects, batch, plan, report_path)


def test_publish_flag_does_not_copy_a_clean_cloud(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "scans"
    scan_root = source_root / "scan-a"
    scan_root.mkdir(parents=True)
    project = scan_root / "scan-a.psx"
    project.write_bytes(b"source")
    projects = tmp_path / "projects.json"
    projects.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_root": str(source_root),
                "projects": [
                    {"scan_id": "scan-a", "project": str(project)}
                ],
            }
        ),
        encoding="utf-8",
    )
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "scan_id": "scan-a",
                        "status": "complete",
                        "output": str(tmp_path / "cleanup" / "scan-a"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "publish.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_tag": "garden-abc123-stride8",
                "scans": [
                    {
                        "scan_id": "scan-a",
                        "action": "publish_flag",
                        "reason": "no coherent plant reconstruction",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    publish_outputs(
        projects,
        batch,
        plan,
        tmp_path / "publish-report.json",
    )

    flag = json.loads(
        (
            scan_root / "plant-cleanup-flag-garden-abc123-stride8.json"
        ).read_text(encoding="utf-8")
    )
    assert flag["reason"] == "no coherent plant reconstruction"
    assert not list(scan_root.glob("plant-cleaned-*.ply"))
