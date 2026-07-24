from __future__ import annotations

import json
from pathlib import Path

import pytest

from railing_removal.quality_report import build_quality_report


def test_quality_report_combines_reference_and_automatic_attention(
    tmp_path: Path,
) -> None:
    scan_output = tmp_path / "cleanup" / "scan-a"
    scan_output.mkdir(parents=True)
    (scan_output / "run-report.json").write_text(
        json.dumps(
            {
                "source_point_count": 50_000,
                "counts": {
                    "floor_candidate": 49_000,
                    "plant_cleaned": 100,
                    "plant_conservative": 1_000,
                    "railing_removed": 0,
                },
                "railing_plan": None,
            }
        ),
        encoding="utf-8",
    )
    batch_report = tmp_path / "batch-report.json"
    batch_report.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "scan_id": "scan-a",
                        "status": "complete",
                        "output": str(scan_output),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scans": {
                    "scan-a": {
                        "status": "no_plant_candidate",
                        "evidence": "reference image contains only pavement",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "quality.json"

    report = build_quality_report(
        batch_report,
        output,
        reference_triage_path=reference,
    )

    result = report["results"][0]
    assert result["review_status"] == "flagged_no_plant_candidate"
    assert {flag["code"] for flag in result["flags"]} >= {
        "reference_no_plant_candidate",
        "very_sparse_source",
        "very_low_retention",
        "strict_conservative_divergence",
    }
    assert report["summary"]["flagged_no_plant_candidate"] == 1

    with pytest.raises(FileExistsError, match="overwrite"):
        build_quality_report(
            batch_report,
            output,
            reference_triage_path=reference,
        )


def test_quality_report_flags_assigned_class_without_model_evidence(
    tmp_path: Path,
) -> None:
    scan_output = tmp_path / "cleanup" / "scan-a"
    scan_output.mkdir(parents=True)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "classes": [
                    {"id": "railing"},
                    {"id": "chain_barrier"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (scan_output / "run-report.json").write_text(
        json.dumps(
            {
                "source_point_count": 10_000,
                "counts": {
                    "floor_candidate": 5_000,
                    "plant_cleaned": 4_000,
                    "plant_conservative": 4_100,
                    "railing_removed": 0,
                },
                "railing_plan": str(plan),
                "railing_completion": {"completed_point_count": 0},
                "scene_evidence": {"fused": {"classes": {}}},
            }
        ),
        encoding="utf-8",
    )
    batch_report = tmp_path / "batch-report.json"
    batch_report.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "scan_id": "scan-a",
                        "status": "complete",
                        "output": str(scan_output),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_quality_report(batch_report, tmp_path / "quality.json")

    missed = [
        flag["object_class"]
        for flag in report["results"][0]["flags"]
        if flag["code"] == "assigned_object_not_detected"
    ]
    assert missed == ["railing", "chain_barrier"]


def test_quality_report_flags_weak_photo_coverage_without_rejecting_scan(
    tmp_path: Path,
) -> None:
    scan_output = tmp_path / "cleanup" / "scan-a"
    scan_output.mkdir(parents=True)
    (scan_output / "run-report.json").write_text(
        json.dumps(
            {
                "source_point_count": 100_000,
                "counts": {
                    "floor_candidate": 90_000,
                    "plant_cleaned": 60_000,
                    "plant_conservative": 61_000,
                },
                "source_photo": {
                    "seen_point_count": 70_000,
                    "unseen_point_count": 30_000,
                    "plant_point_count": 500,
                    "background_point_count": 69_500,
                },
            }
        ),
        encoding="utf-8",
    )
    batch_report = tmp_path / "batch-report.json"
    batch_report.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "scan_id": "scan-a",
                        "status": "complete",
                        "output": str(scan_output),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_quality_report(batch_report, tmp_path / "quality.json")

    result = report["results"][0]
    assert result["review_status"] == "needs_review"
    assert result["metrics"]["source_photo_unseen_ratio"] == 0.3
    assert {flag["code"] for flag in result["flags"]} >= {
        "weak_source_photo_coverage",
        "low_source_photo_plant_evidence",
    }
