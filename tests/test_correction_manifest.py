import json
from pathlib import Path

import pytest

from railing_removal.correction_manifest import (
    build_correction_manifest,
    build_targeted_correction_manifest,
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
        "scene_plan": "F:\\plans\\partial.json",
        "review_artifacts": False,
    }
    assert result["scans"][1]["scene_plan"] == "F:\\plans\\failed.json"
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


def test_builds_targeted_visual_qa_corrections_with_object_scene_plans(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "cleanup-manifest.json"
    scene_catalog = tmp_path / "scene-catalog.json"
    assignments = tmp_path / "visual-qa-assignments.json"
    output = tmp_path / "targeted-corrections.json"
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "scans": [
                {
                    "scan_id": "scan-with-turf",
                    "source": "F:\\clouds\\scan-with-turf.ply",
                    "config": "F:\\configs\\scan-with-turf.json",
                },
                {
                    "scan_id": "accepted-scan",
                    "source": "F:\\clouds\\accepted-scan.ply",
                    "config": "F:\\configs\\accepted-scan.json",
                },
            ],
        },
    )
    _write_json(
        scene_catalog,
        {
            "schema_version": 1,
            "target_intent": "Keep plants and remove assigned objects.",
            "plant_prompt": "complete living plants including roots",
            "classes": {
                "turf_ground": {
                    "prompt": "flat lawn turf ground surface",
                    "distractor_prompt": "raised plants roots and leaves",
                    "anchor_strategy": "semantic",
                    "decision_policy": "ground_surface",
                }
            },
            "assignments": {},
        },
    )
    _write_json(
        assignments,
        {
            "schema_version": 1,
            "assignments": {"scan-with-turf": ["turf_ground"]},
        },
    )

    result = build_targeted_correction_manifest(
        source_manifest,
        scene_catalog,
        assignments,
        output,
    )

    assert [scan["scan_id"] for scan in result["scans"]] == [
        "scan-with-turf"
    ]
    correction = result["scans"][0]
    assert correction["review_artifacts"] is False
    plan = json.loads(
        Path(correction["scene_plan"]).read_text(encoding="utf-8")
    )
    assert plan["scan_id"] == "scan-with-turf"
    assert [item["id"] for item in plan["classes"]] == ["turf_ground"]
    assert plan["classes"][0]["decision_policy"] == "ground_surface"
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_final_reference_review_assigns_only_verified_ground_and_structure(
) -> None:
    repository = Path(__file__).resolve().parents[1]
    catalog = json.loads(
        (
            repository
            / "configs"
            / "scene-plan-catalog-202607-sf.json"
        ).read_text(encoding="utf-8")
    )
    corrections = json.loads(
        (
            repository
            / "configs"
            / "visual-qa-corrections-reference-review-202607-sf.json"
        ).read_text(encoding="utf-8")
    )
    retry = json.loads(
        (
            repository
            / "configs"
            / "visual-qa-corrections-reference-retry-202607-sf.json"
        ).read_text(encoding="utf-8")
    )

    assert catalog["classes"]["fence"]["completion_strategy"] == "rigid_surface"
    assert corrections["assignments"] == {
        "2026-07-15 13.31.42": ["turf_ground"],
        "2026-07-16 09.44.32": ["turf_ground"],
        "2026-07-16 13.19.49": ["turf_ground"],
        "2026-07-16 13.20.14": ["turf_ground"],
        "2026-07-17 12.17.25": ["turf_ground", "fence"],
        "2026-07-17 12.21.21": ["turf_ground"],
    }
    assert set(corrections["assignments"]).isdisjoint(
        {
            "2026-07-15 17.26.29",
            "2026-07-16 09.39.40",
            "2026-07-17 11.05.52",
            "2026-07-17 11.32.39",
            "2026-07-17 11.54.07",
            "2026-07-17 12.55.23",
        }
    )
    assert retry["assignments"] == {
        "2026-07-16 09.44.32": ["turf_ground"],
        "2026-07-17 12.17.25": ["turf_ground", "fence"],
    }
