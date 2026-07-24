from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"invalid schema_version in {path}")
    return value


def build_correction_manifest(
    source_manifest_path: Path,
    baseline_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select all unfinished baseline scans without legacy scene plans."""
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    source_manifest = _read_object(source_manifest_path)
    source_scans = source_manifest.get("scans")
    if not isinstance(source_scans, list):
        raise ValueError("source manifest requires scans")
    by_id = {
        str(scan.get("scan_id", "")).strip(): scan
        for scan in source_scans
        if isinstance(scan, dict)
    }

    baseline_report = _read_object(baseline_report_path)
    results = baseline_report.get("results")
    if not isinstance(results, list):
        raise ValueError("baseline report requires results")

    corrections: list[dict[str, str]] = []
    for result in results:
        if not isinstance(result, dict) or result.get("status") == "complete":
            continue
        scan_id = str(result.get("scan_id", "")).strip()
        source = by_id.get(scan_id)
        if source is None:
            raise ValueError(
                f"baseline report references unknown scan: {scan_id}"
            )
        corrections.append(
            {
                "scan_id": scan_id,
                "source": str(source["source"]),
                "config": str(source["config"]),
            }
        )
    if not corrections:
        raise ValueError("baseline report has no scans requiring correction")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "scans": corrections,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return manifest
