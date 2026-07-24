from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from railing_removal.photo_review import build_reference_review


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 120), color).save(path)


def test_reference_review_samples_photos_without_changing_sources(
    tmp_path: Path,
) -> None:
    photos = tmp_path / "photos" / "scan-a"
    for index in range(7):
        _image(photos / "png" / f"{index:05d}.png", (index, 50, 90))
    before = {
        path: path.read_bytes()
        for path in photos.rglob("*.png")
    }
    manifest = tmp_path / "projects.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {
                        "scan_id": "scan-a",
                        "project": str(tmp_path / "scan-a.psx"),
                        "photo_dir": str(photos),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_reference_review(
        manifest,
        tmp_path / "review",
        frames_per_scan=3,
        scans_per_page=1,
    )

    assert report["summary"] == {"complete": 1, "missing_photos": 0}
    assert (tmp_path / "review" / "scans" / "scan-a.jpg").is_file()
    assert (tmp_path / "review" / "pages" / "page-001.jpg").is_file()
    assert {
        path: path.read_bytes()
        for path in photos.rglob("*.png")
    } == before


def test_reference_review_records_missing_photos_and_never_overwrites(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "projects.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {
                        "scan_id": "scan-missing",
                        "project": str(tmp_path / "scan-missing.psx"),
                        "photo_dir": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "review"
    first = build_reference_review(manifest, output)

    assert first["summary"] == {"complete": 0, "missing_photos": 1}
    preserved = (output / "scans" / "scan-missing.jpg").read_bytes()
    try:
        build_reference_review(manifest, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("finalized review must not be overwritten")
    assert (output / "scans" / "scan-missing.jpg").read_bytes() == preserved
