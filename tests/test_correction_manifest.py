import json
from pathlib import Path

import pytest

from railing_removal.correction_manifest import (
    build_correction_manifest,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_builds_additive_manifest_for_every_noncomplete_baseline_scan(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "cleanup-manifest.json"
    baseline_report = tmp_path / "batch-report.json"
    output = tmp_path / "corrections.json"
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "scans": [
                {
                    "scan_id": "complete",
                    "source": "F:\\clouds\\complete.ply",
                    "config": "F:\\configs\\complete.json",
                    "scene_plan": "F:\\plans\\complete.json",
                },
                {
                    "scan_id": "partial",
                    "source": "F:\\clouds\\partial.ply",
                    "config": "F:\\configs\\partial.json",
                    "scene_plan": "F:\\plans\\partial.json",
                },
                {
                    "scan_id": "failed",
                    "source": "F:\\clouds\\failed.ply",
                    "config": "F:\\configs\\failed.json",
                    "scene_plan": "F:\\plans\\failed.json",
                },
            ],
        },
    )
    _write_json(
        baseline_report,
        {
            "schema_version": 1,
            "results": [
                {"scan_id": "complete", "status": "complete"},
                {"scan_id": "partial", "status": "partial"},
                {"scan_id": "failed", "status": "failed"},
            ],
        },
    )

    result = build_correction_manifest(
        source_manifest,
        baseline_report,
        output,
    )

    assert [scan["scan_id"] for scan in result["scans"]] == [
        "partial",
        "failed",
    ]
    assert result["scans"][0] == {
        "scan_id": "partial",
        "source": "F:\\clouds\\partial.ply",
        "config": "F:\\configs\\partial.json",
    }
    assert "scene_plan" not in result["scans"][1]
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_refuses_to_overwrite_or_omit_an_unknown_failed_scan(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "cleanup-manifest.json"
    baseline_report = tmp_path / "batch-report.json"
    output = tmp_path / "corrections.json"
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "scans": [
                {
                    "scan_id": "known",
                    "source": "F:\\known.ply",
                    "config": "F:\\known.json",
                }
            ],
        },
    )
    _write_json(
        baseline_report,
        {
            "schema_version": 1,
            "results": [{"scan_id": "unknown", "status": "failed"}],
        },
    )

    with pytest.raises(ValueError, match="unknown"):
        build_correction_manifest(
            source_manifest,
            baseline_report,
            output,
        )

    output.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_correction_manifest(
            source_manifest,
            baseline_report,
            output,
        )
    assert output.read_text(encoding="utf-8") == "preserve me"
