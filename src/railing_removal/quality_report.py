from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def _flag(code: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "reason": reason, **extra}


def _load_reference_triage(
    path: Path | None,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    scans = value.get("scans")
    if value.get("schema_version") != 1 or not isinstance(scans, dict):
        raise ValueError("invalid reference triage")
    return scans


def _assigned_object_flags(run_report: dict[str, Any]) -> list[dict[str, Any]]:
    plan_path_value = run_report.get("railing_plan")
    if not plan_path_value:
        return []
    plan_path = Path(plan_path_value).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    flags: list[dict[str, Any]] = []
    fused = (
        run_report.get("scene_evidence", {})
        .get("fused", {})
        .get("classes", {})
    )
    for object_class in plan.get("classes", []):
        class_id = str(object_class["id"])
        if class_id == "railing":
            detected = int(
                run_report.get("railing_completion", {}).get(
                    "completed_point_count",
                    0,
                )
            )
        else:
            detected = int(
                fused.get(class_id, {}).get("voted_point_count", 0)
            )
        if detected == 0:
            flags.append(
                _flag(
                    "assigned_object_not_detected",
                    "The scan-specific object class produced no removal "
                    "evidence and needs visual verification.",
                    object_class=class_id,
                )
            )
    return flags


def _complete_scan_quality(
    result: dict[str, Any],
    reference: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scan_id = str(result["scan_id"])
    run_report_path = Path(result["output"]) / "run-report.json"
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    source_count = int(run_report["source_point_count"])
    counts = run_report["counts"]
    strict_count = int(counts["plant_cleaned"])
    conservative_count = int(counts["plant_conservative"])
    floor_count = int(counts["floor_candidate"])
    retention_ratio = strict_count / source_count if source_count else 0.0
    conservative_gap = (
        (conservative_count - strict_count) / conservative_count
        if conservative_count
        else 0.0
    )
    flags: list[dict[str, Any]] = []
    reference_value = reference.get(scan_id)
    if (
        reference_value
        and reference_value.get("status") == "no_plant_candidate"
    ):
        flags.append(
            _flag(
                "reference_no_plant_candidate",
                str(reference_value.get("evidence", "")).strip(),
            )
        )
    elif reference_value:
        flags.append(
            _flag(
                "reference_manual_review",
                str(reference_value.get("evidence", "")).strip(),
            )
        )
    if source_count < 100_000:
        flags.append(
            _flag(
                "very_sparse_source",
                "The canonical source contains fewer than 100,000 points.",
            )
        )
    if strict_count < 1_000:
        flags.append(
            _flag(
                "empty_or_nearly_empty_cleanup",
                "The strict cleaned cloud contains fewer than 1,000 points.",
            )
        )
    if retention_ratio < 0.01:
        flags.append(
            _flag(
                "very_low_retention",
                "The strict plant cloud retains less than 1% of the source.",
            )
        )
    if conservative_gap > 0.20:
        flags.append(
            _flag(
                "strict_conservative_divergence",
                "Strict and conservative plant layers differ by more than "
                "20%; inspect roots and boundary leaves.",
            )
        )
    flags.extend(_assigned_object_flags(run_report))
    if any(
        flag["code"] == "reference_no_plant_candidate" for flag in flags
    ):
        review_status = "flagged_no_plant_candidate"
    elif flags:
        review_status = "needs_review"
    else:
        review_status = "complete"
    return {
        "scan_id": scan_id,
        "processing_status": "complete",
        "review_status": review_status,
        "metrics": {
            "source_point_count": source_count,
            "plant_cleaned": strict_count,
            "plant_conservative": conservative_count,
            "floor_candidate": floor_count,
            "retention_ratio": retention_ratio,
            "strict_conservative_gap_ratio": conservative_gap,
        },
        "flags": flags,
        "output": str(Path(result["output"]).resolve()),
    }


def build_quality_report(
    batch_report_path: Path,
    output: Path,
    *,
    reference_triage_path: Path | None = None,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite quality report: {output}")
    batch_report = json.loads(
        batch_report_path.resolve().read_text(encoding="utf-8")
    )
    reference = _load_reference_triage(reference_triage_path)
    results: list[dict[str, Any]] = []
    for result in batch_report["results"]:
        if result["status"] == "complete":
            results.append(_complete_scan_quality(result, reference))
        else:
            results.append(
                {
                    "scan_id": str(result["scan_id"]),
                    "processing_status": str(result["status"]),
                    "review_status": "processing_failure",
                    "metrics": {},
                    "flags": [
                        _flag(
                            "processing_failure",
                            str(result.get("error", result["status"])),
                        )
                    ],
                    "output": str(result["output"]),
                }
            )
    statuses = (
        "complete",
        "needs_review",
        "flagged_no_plant_candidate",
        "processing_failure",
    )
    report = {
        "schema_version": 1,
        "batch_report": str(batch_report_path.resolve()),
        "reference_triage": (
            str(reference_triage_path.resolve())
            if reference_triage_path is not None
            else None
        ),
        "summary": {
            status: sum(
                result["review_status"] == status for result in results
            )
            for status in statuses
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial-{uuid4().hex}")
    with partial.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    partial.rename(output)
    return report
