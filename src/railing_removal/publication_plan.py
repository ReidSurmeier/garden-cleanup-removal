from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from railing_removal.publish_outputs import ARTIFACT_TAG


def _read_object(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"invalid schema version: {resolved}")
    return value


def _unique_ids(
    values: object,
    *,
    document: str,
) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{document} requires a nonempty list")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, dict):
            raise ValueError(f"{document} contains an invalid entry")
        scan_id = str(item.get("scan_id", "")).strip()
        if not scan_id or scan_id in seen or Path(scan_id).name != scan_id:
            raise ValueError(f"{document} contains an unsafe scan ID")
        seen.add(scan_id)
        result.append(scan_id)
    return result


def build_publication_plan(
    project_manifest_path: Path,
    batch_report_path: Path,
    reference_triage_path: Path,
    output_path: Path,
    *,
    artifact_tag: str,
) -> dict[str, Any]:
    """Build an exhaustive clean-or-flag plan for the canonical inventory."""
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if ARTIFACT_TAG.fullmatch(artifact_tag) is None:
        raise ValueError("invalid artifact tag")

    project_manifest = _read_object(project_manifest_path)
    project_ids = _unique_ids(
        project_manifest.get("projects"),
        document="project manifest",
    )
    batch_report = _read_object(batch_report_path)
    batch_results = batch_report.get("results")
    batch_ids = _unique_ids(batch_results, document="batch report")
    if set(batch_ids) != set(project_ids):
        raise ValueError("batch report does not exactly match project inventory")
    status_by_id = {
        str(item["scan_id"]): item.get("status")
        for item in batch_results
    }
    incomplete = [
        scan_id
        for scan_id in project_ids
        if status_by_id[scan_id] != "complete"
    ]
    if incomplete:
        raise ValueError(f"batch scan is not complete: {incomplete[0]}")

    triage_document = _read_object(reference_triage_path)
    triage = triage_document.get("scans")
    if not isinstance(triage, dict):
        raise ValueError("reference triage requires scans")
    unknown = set(triage) - set(project_ids)
    if unknown:
        raise ValueError(
            f"triage references unknown scan: {sorted(unknown)[0]}"
        )

    plan_scans: list[dict[str, str]] = []
    for scan_id in project_ids:
        decision = triage.get(scan_id)
        if decision is None:
            plan_scans.append(
                {
                    "scan_id": scan_id,
                    "action": "publish_clean",
                }
            )
            continue
        if not isinstance(decision, dict):
            raise ValueError(f"invalid triage decision: {scan_id}")
        status = str(decision.get("status", "")).strip()
        evidence = str(decision.get("evidence", "")).strip()
        if not evidence:
            raise ValueError(f"triage evidence is missing: {scan_id}")
        if status == "accept_clean":
            plan_scans.append(
                {
                    "scan_id": scan_id,
                    "action": "publish_clean",
                }
            )
        elif status in {
            "no_plant_candidate",
            "no_coherent_reconstruction",
            "manual_cleanup_required",
        }:
            flag = {
                "scan_id": scan_id,
                "action": "publish_flag",
                "reason": evidence,
            }
            if status == "manual_cleanup_required":
                flag["flag_status"] = status
            plan_scans.append(flag)
        else:
            raise ValueError(
                f"unresolved reference triage status for {scan_id}: {status}"
            )

    summary = {
        action: sum(item["action"] == action for item in plan_scans)
        for action in ("publish_clean", "publish_flag")
    }
    plan = {
        "schema_version": 1,
        "artifact_tag": artifact_tag,
        "project_manifest": str(project_manifest_path.resolve()),
        "batch_report": str(batch_report_path.resolve()),
        "reference_triage": str(reference_triage_path.resolve()),
        "summary": summary,
        "scans": plan_scans,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, indent=2, sort_keys=True)
        output.write("\n")
    return plan
