from __future__ import annotations

import json
from pathlib import Path

import pytest

from railing_removal.blender_orientation_production import (
    PRODUCTION_BLEND_FILENAME,
    build_production_orientation_manifest,
)


def test_production_manifest_plans_one_new_blend_beside_each_cleaned_ply(
    tmp_path: Path,
) -> None:
    production_root = tmp_path / "scans"
    available = production_root / "scan-a"
    missing = production_root / "scan-b"
    available.mkdir(parents=True)
    missing.mkdir()
    source = available / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
    source.write_bytes(b"immutable cleaned cloud")
    for scan_dir in (available, missing):
        (scan_dir / f"{scan_dir.name}.psx").write_text(
            "immutable project",
            encoding="utf-8",
        )
    projects = tmp_path / "projects.json"
    projects.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {
                        "scan_id": scan_dir.name,
                        "project": str(scan_dir / f"{scan_dir.name}.psx"),
                    }
                    for scan_dir in (available, missing)
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "production-manifest.json"

    manifest = build_production_orientation_manifest(
        projects,
        production_root,
        output,
    )

    assert manifest["summary"] == {
        "eligible": 1,
        "missing_cleaned_ply": 1,
    }
    assert manifest["scans"] == [
        {
            "scan_id": "scan-a",
            "source": str(source.resolve()),
            "source_opened_read_only": True,
            "blend_output": str(
                (available / PRODUCTION_BLEND_FILENAME).resolve()
            ),
        }
    ]
    assert manifest["missing"] == [
        {
            "scan_id": "scan-b",
            "reason": "missing_cleaned_ply",
            "expected_source": str(
                (
                    missing
                    / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
                ).resolve()
            ),
        }
    ]
    assert output.is_file()
    assert source.read_bytes() == b"immutable cleaned cloud"
    assert not (available / PRODUCTION_BLEND_FILENAME).exists()


def test_production_manifest_never_plans_over_an_existing_blend(
    tmp_path: Path,
) -> None:
    production_root = tmp_path / "scans"
    scan_dir = production_root / "scan-a"
    scan_dir.mkdir(parents=True)
    project = scan_dir / "scan-a.psx"
    project.write_text("immutable project", encoding="utf-8")
    (scan_dir / "plant-cleaned-garden-ec2fbd1-final-v2.ply").write_bytes(
        b"immutable cleaned cloud"
    )
    existing = scan_dir / PRODUCTION_BLEND_FILENAME
    existing.write_bytes(b"existing review")
    projects = tmp_path / "projects.json"
    projects.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {"scan_id": "scan-a", "project": str(project)}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="existing production Blender review",
    ):
        build_production_orientation_manifest(
            projects,
            production_root,
            tmp_path / "production-manifest.json",
        )

    assert existing.read_bytes() == b"existing review"


def test_production_manifest_rejects_a_linked_cleaned_ply(
    tmp_path: Path,
) -> None:
    production_root = tmp_path / "scans"
    scan_dir = production_root / "scan-a"
    scan_dir.mkdir(parents=True)
    project = scan_dir / "scan-a.psx"
    project.write_text("immutable project", encoding="utf-8")
    outside = tmp_path / "outside.ply"
    outside.write_bytes(b"outside source")
    (
        scan_dir / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
    ).symlink_to(outside)
    projects = tmp_path / "projects.json"
    projects.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {"scan_id": "scan-a", "project": str(project)}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="linked cleaned PLY"):
        build_production_orientation_manifest(
            projects,
            production_root,
            tmp_path / "production-manifest.json",
        )
