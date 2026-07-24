import json
from pathlib import Path

import pytest

from railing_removal.publication_plan import build_publication_plan


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_reference_failure_becomes_a_flag_and_other_scans_publish_clean(
    tmp_path: Path,
) -> None:
    projects = _write(
        tmp_path / "projects.json",
        {
            "schema_version": 1,
            "projects": [
                {"scan_id": "scan-a"},
                {"scan_id": "scan-b"},
            ],
        },
    )
    batch = _write(
        tmp_path / "batch.json",
        {
            "schema_version": 1,
            "results": [
                {"scan_id": "scan-a", "status": "complete"},
                {"scan_id": "scan-b", "status": "complete"},
            ],
        },
    )
    triage = _write(
        tmp_path / "triage.json",
        {
            "schema_version": 1,
            "scans": {
                "scan-b": {
                    "status": "no_coherent_reconstruction",
                    "evidence": "Reference plant did not reconstruct.",
                }
            },
        },
    )

    plan = build_publication_plan(
        projects,
        batch,
        triage,
        tmp_path / "publication.json",
        artifact_tag="clean-v1",
    )

    assert plan["scans"] == [
        {
            "scan_id": "scan-a",
            "action": "publish_clean",
        },
        {
            "scan_id": "scan-b",
            "action": "publish_flag",
            "reason": "Reference plant did not reconstruct.",
        },
    ]
    assert plan["summary"] == {
        "publish_clean": 1,
        "publish_flag": 1,
    }


def test_publication_plan_refuses_an_incomplete_inventory(
    tmp_path: Path,
) -> None:
    projects = _write(
        tmp_path / "projects.json",
        {
            "schema_version": 1,
            "projects": [
                {"scan_id": "scan-a"},
                {"scan_id": "scan-b"},
            ],
        },
    )
    batch = _write(
        tmp_path / "batch.json",
        {
            "schema_version": 1,
            "results": [
                {"scan_id": "scan-a", "status": "complete"},
            ],
        },
    )
    triage = _write(
        tmp_path / "triage.json",
        {"schema_version": 1, "scans": {}},
    )
    output = tmp_path / "publication.json"

    with pytest.raises(
        ValueError,
        match="does not exactly match project inventory",
    ):
        build_publication_plan(
            projects,
            batch,
            triage,
            output,
            artifact_tag="clean-v1",
        )

    assert not output.exists()


def test_unresolved_manmade_structure_becomes_a_manual_cleanup_flag(
    tmp_path: Path,
) -> None:
    projects = _write(
        tmp_path / "projects.json",
        {
            "schema_version": 1,
            "projects": [{"scan_id": "scan-a"}],
        },
    )
    batch = _write(
        tmp_path / "batch.json",
        {
            "schema_version": 1,
            "results": [{"scan_id": "scan-a", "status": "complete"}],
        },
    )
    triage = _write(
        tmp_path / "triage.json",
        {
            "schema_version": 1,
            "scans": {
                "scan-a": {
                    "status": "manual_cleanup_required",
                    "evidence": "Pavilion remains interleaved with plants.",
                }
            },
        },
    )

    plan = build_publication_plan(
        projects,
        batch,
        triage,
        tmp_path / "publication.json",
        artifact_tag="clean-v1",
    )

    assert plan["scans"] == [
        {
            "scan_id": "scan-a",
            "action": "publish_flag",
            "flag_status": "manual_cleanup_required",
            "reason": "Pavilion remains interleaved with plants.",
        }
    ]


def test_publication_plan_refuses_unresolved_reference_triage(
    tmp_path: Path,
) -> None:
    projects = _write(
        tmp_path / "projects.json",
        {
            "schema_version": 1,
            "projects": [{"scan_id": "scan-a"}],
        },
    )
    batch = _write(
        tmp_path / "batch.json",
        {
            "schema_version": 1,
            "results": [
                {"scan_id": "scan-a", "status": "complete"},
            ],
        },
    )
    triage = _write(
        tmp_path / "triage.json",
        {
            "schema_version": 1,
            "scans": {
                "scan-a": {
                    "status": "manual_review",
                    "evidence": "This scan still requires a decision.",
                }
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="unresolved reference triage status",
    ):
        build_publication_plan(
            projects,
            batch,
            triage,
            tmp_path / "publication.json",
            artifact_tag="clean-v1",
        )
