from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> list[dict[str, Any]]:
    resolved = path.resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    results = value.get("results")
    if value.get("schema_version") != 1 or not isinstance(results, list):
        raise ValueError(f"invalid batch report: {resolved}")
    seen: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            raise ValueError(f"invalid batch result in {resolved}")
        scan_id = str(item.get("scan_id", "")).strip()
        if not scan_id or scan_id in seen:
            raise ValueError(f"unsafe or duplicate scan ID in {resolved}")
        seen.add(scan_id)
        loaded.append(dict(item))
    if not loaded:
        raise ValueError(f"empty batch report: {resolved}")
    return loaded


def build_batch_overlay(
    report_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Select the latest reviewed complete result for each base scan."""
    if not report_paths:
        raise ValueError("at least one batch report is required")
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    resolved_reports = [path.resolve() for path in report_paths]
    base_results = _load_results(resolved_reports[0])
    order = [str(item["scan_id"]) for item in base_results]
    selected = {
        scan_id: {
            **item,
            "selected_report": str(resolved_reports[0]),
        }
        for scan_id, item in zip(order, base_results, strict=True)
    }
    for report_path in resolved_reports[1:]:
        for item in _load_results(report_path):
            scan_id = str(item["scan_id"])
            if scan_id not in selected:
                raise ValueError(
                    f"override contains unknown base scan: {scan_id}"
                )
            if item.get("status") != "complete":
                raise ValueError(
                    f"override result is not complete: {scan_id}"
                )
            selected[scan_id] = {
                **item,
                "selected_report": str(report_path),
            }

    results = [selected[scan_id] for scan_id in order]
    summary = {
        status: sum(item.get("status") == status for item in results)
        for status in ("complete", "failed", "partial")
    }
    overlay = {
        "schema_version": 1,
        "strategy": "ordered-reviewed-batch-overlay-v1",
        "source_reports": [str(path) for path in resolved_reports],
        "output_root": str(output_path.parent),
        "summary": summary,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(overlay, output, indent=2, sort_keys=True)
        output.write("\n")
    return overlay
