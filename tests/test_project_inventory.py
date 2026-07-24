from __future__ import annotations

from pathlib import Path

import pytest

from railing_removal.project_inventory import build_project_manifest


def test_inventory_is_sorted_scoped_and_links_matching_photos(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "canonical"
    photo_root = tmp_path / "photos"
    for scan_id in ("scan-b", "scan-a"):
        project_dir = source_root / scan_id
        project_dir.mkdir(parents=True)
        (project_dir / f"{scan_id}.psx").write_text(
            "source",
            encoding="utf-8",
        )
    (photo_root / "scan-a").mkdir(parents=True)
    backup = tmp_path / "backup" / "scan-hidden"
    backup.mkdir(parents=True)
    (backup / "scan-hidden.psx").write_text("backup", encoding="utf-8")
    output = tmp_path / "manifest.json"

    manifest = build_project_manifest(
        source_root,
        output,
        photo_root=photo_root,
    )

    assert [item["scan_id"] for item in manifest["projects"]] == [
        "scan-a",
        "scan-b",
    ]
    assert len(manifest["projects"]) == 2
    assert manifest["projects"][0]["photo_dir"] == str(
        (photo_root / "scan-a").resolve()
    )
    assert manifest["projects"][1]["photo_dir"] is None
    assert "backup" not in output.read_text(encoding="utf-8")


def test_inventory_refuses_to_overwrite_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "canonical"
    project_dir = source_root / "scan-a"
    project_dir.mkdir(parents=True)
    (project_dir / "scan-a.psx").write_text("source", encoding="utf-8")
    output = tmp_path / "manifest.json"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_project_manifest(source_root, output)

    assert output.read_text(encoding="utf-8") == "keep"
