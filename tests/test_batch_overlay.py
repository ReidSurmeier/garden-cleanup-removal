import json
from pathlib import Path

import pytest

from railing_removal.batch_overlay import build_batch_overlay


def _write_report(
    path: Path,
    results: list[dict[str, object]],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "results": results,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_later_complete_reviewed_result_replaces_the_base_scan(
    tmp_path: Path,
) -> None:
    base = _write_report(
        tmp_path / "base.json",
        [
            {
                "scan_id": "scan-a",
                "status": "complete",
                "output": "base/a",
            },
            {
                "scan_id": "scan-b",
                "status": "complete",
                "output": "base/b",
            },
        ],
    )
    correction = _write_report(
        tmp_path / "correction.json",
        [
            {
                "scan_id": "scan-b",
                "status": "complete",
                "output": "corrected/b",
            }
        ],
    )

    overlay = build_batch_overlay(
        [base, correction],
        tmp_path / "overlay.json",
    )

    assert [item["scan_id"] for item in overlay["results"]] == [
        "scan-a",
        "scan-b",
    ]
    assert overlay["results"][1]["output"] == "corrected/b"
    assert overlay["results"][1]["selected_report"] == str(
        correction.resolve()
    )
    assert overlay["summary"] == {
        "complete": 2,
        "failed": 0,
        "partial": 0,
    }


def test_override_cannot_introduce_a_scan_outside_the_base_inventory(
    tmp_path: Path,
) -> None:
    base = _write_report(
        tmp_path / "base.json",
        [
            {
                "scan_id": "scan-a",
                "status": "complete",
                "output": "base/a",
            }
        ],
    )
    foreign = _write_report(
        tmp_path / "foreign.json",
        [
            {
                "scan_id": "scan-outside-root",
                "status": "complete",
                "output": "foreign/scan",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="override contains unknown base scan",
    ):
        build_batch_overlay(
            [base, foreign],
            tmp_path / "overlay.json",
        )
