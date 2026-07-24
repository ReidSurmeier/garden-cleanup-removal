from __future__ import annotations

import json
from pathlib import Path

from plant_cleanup.source_photo_cleanup_batch import (
    run_source_photo_cleanup_batch,
)


def test_source_photo_cleanup_batch_is_additive_and_resumable(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "projects.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {"scan_id": "scan-1", "project": "one.psx"},
                    {"scan_id": "scan-2", "project": "two.psx"},
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline"
    semantic = tmp_path / "semantic"
    output = tmp_path / "cleanup"
    for scan_id in ("scan-1", "scan-2"):
        baseline_scan = baseline / scan_id
        semantic_scan = semantic / scan_id
        baseline_scan.mkdir(parents=True)
        semantic_scan.mkdir(parents=True)
        (baseline_scan / "run-report.json").write_text(
            json.dumps({"source_opened_read_only": True}),
            encoding="utf-8",
        )
        (semantic_scan / "report.json").write_text(
            json.dumps({"project_opened_read_only": True}),
            encoding="utf-8",
        )
    complete = output / "scan-1"
    complete.mkdir(parents=True)
    (complete / "run-report.json").write_text(
        json.dumps(
            {
                "source_opened_read_only": True,
                "counts": {"plant_cleaned": 10},
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[Path, Path, Path]] = []

    def cleanup(
        baseline_scan: Path,
        semantic_scan: Path,
        destination: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append((baseline_scan, semantic_scan, destination))
        destination.mkdir()
        report = {
            "source_opened_read_only": True,
            "counts": {"plant_cleaned": 20},
        }
        (destination / "run-report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return report

    report = run_source_photo_cleanup_batch(
        manifest,
        baseline,
        semantic,
        output,
        cleanup=cleanup,
    )

    assert calls == [
        (
            (baseline / "scan-2").resolve(),
            (semantic / "scan-2").resolve(),
            (output / "scan-2").resolve(),
        )
    ]
    assert report["summary"] == {
        "complete": 2,
        "partial": 0,
        "failed": 0,
    }


def test_source_photo_cleanup_batch_preserves_partial_output(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "projects.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {"scan_id": "scan-1", "project": "one.psx"}
                ],
            }
        ),
        encoding="utf-8",
    )
    partial = tmp_path / "cleanup" / "scan-1"
    partial.mkdir(parents=True)
    evidence = partial / "evidence.txt"
    evidence.write_text("preserve", encoding="utf-8")

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("partial output must not be processed")

    report = run_source_photo_cleanup_batch(
        manifest,
        tmp_path / "baseline",
        tmp_path / "semantic",
        tmp_path / "cleanup",
        cleanup=forbidden,
    )

    assert report["summary"] == {
        "complete": 0,
        "partial": 1,
        "failed": 0,
    }
    assert evidence.read_text(encoding="utf-8") == "preserve"
