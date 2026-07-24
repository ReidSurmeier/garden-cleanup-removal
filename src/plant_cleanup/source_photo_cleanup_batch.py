from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


Cleanup = Callable[..., dict[str, Any]]
Progress = Callable[[str], None]


def _scan_ids(path: Path) -> list[str]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    items = value.get("projects", value.get("scans"))
    if value.get("schema_version") != 1 or not isinstance(items, list):
        raise ValueError("invalid project or targeted correction manifest")
    scan_ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        scan_id = str(item.get("scan_id", "")).strip()
        if not scan_id or scan_id in seen:
            raise ValueError("scan IDs must be nonempty and unique")
        seen.add(scan_id)
        scan_ids.append(scan_id)
    if not scan_ids:
        raise ValueError("project manifest requires scans")
    return scan_ids


def _validate_report(report: dict[str, Any]) -> None:
    if not report.get("source_opened_read_only"):
        raise ValueError("cleanup report lacks read-only provenance")
    if not isinstance(report.get("counts"), dict):
        raise ValueError("cleanup report lacks output counts")


def run_source_photo_cleanup_batch(
    manifest_path: Path,
    baseline_root: Path,
    source_photo_root: Path,
    output_root: Path,
    *,
    baseline_correction_root: Path | None = None,
    cleanup: Cleanup | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    if cleanup is None:
        from plant_cleanup.source_photo_cleanup import (
            run_source_photo_cleanup,
        )

        cleanup = run_source_photo_cleanup
    scan_ids = _scan_ids(manifest_path)
    baseline_root = baseline_root.resolve()
    baseline_correction_root = (
        baseline_correction_root.resolve()
        if baseline_correction_root is not None
        else None
    )
    source_photo_root = source_photo_root.resolve()
    output_root = output_root.resolve()
    batch_report_path = output_root / "batch-report.json"
    if batch_report_path.exists():
        raise FileExistsError(
            f"source-photo cleanup batch is already finalized: "
            f"{batch_report_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda _: None)
    results: list[dict[str, Any]] = []
    for index, scan_id in enumerate(scan_ids, start=1):
        destination = output_root / scan_id
        report_path = destination / "run-report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _validate_report(report)
            progress(
                f"[{index}/{len(scan_ids)}] already complete {scan_id}"
            )
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "complete",
                    "counts": report["counts"],
                    "output": str(destination),
                }
            )
            continue
        if destination.exists():
            progress(f"[{index}/{len(scan_ids)}] partial output {scan_id}")
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "partial",
                    "output": str(destination),
                }
            )
            continue
        corrected_scan = (
            baseline_correction_root / scan_id
            if baseline_correction_root is not None
            else None
        )
        baseline_scan = (
            corrected_scan
            if corrected_scan is not None
            and (corrected_scan / "run-report.json").is_file()
            else baseline_root / scan_id
        )
        source_photo_scan = source_photo_root / scan_id
        progress(f"[{index}/{len(scan_ids)}] processing {scan_id}")
        try:
            baseline_report_path = baseline_scan / "run-report.json"
            source_photo_report_path = source_photo_scan / "report.json"
            for dependency in (
                baseline_report_path,
                source_photo_report_path,
            ):
                if not dependency.is_file():
                    raise FileNotFoundError(dependency)
            baseline_report = json.loads(
                baseline_report_path.read_text(encoding="utf-8")
            )
            source_photo_report = json.loads(
                source_photo_report_path.read_text(encoding="utf-8")
            )
            if not baseline_report.get("source_opened_read_only"):
                raise ValueError("baseline lacks read-only provenance")
            if not source_photo_report.get("project_opened_read_only"):
                raise ValueError(
                    "source-photo evidence lacks read-only provenance"
                )
            report = cleanup(
                baseline_scan.resolve(),
                source_photo_scan.resolve(),
                destination,
                progress=lambda stage, prefix=scan_id: progress(
                    f"[{prefix}] {stage}"
                ),
            )
            _validate_report(report)
            if not report_path.is_file():
                raise FileNotFoundError(report_path)
        except Exception as error:
            progress(f"[{index}/{len(scan_ids)}] failed {scan_id}: {error}")
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "output": str(destination),
                }
            )
            continue
        progress(f"[{index}/{len(scan_ids)}] complete {scan_id}")
        results.append(
            {
                "scan_id": scan_id,
                "status": "complete",
                "counts": report["counts"],
                "output": str(destination),
            }
        )
    summary = {
        status: sum(result["status"] == status for result in results)
        for status in ("complete", "partial", "failed")
    }
    batch_report = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "baseline_root": str(baseline_root),
        "baseline_correction_root": (
            str(baseline_correction_root)
            if baseline_correction_root is not None
            else None
        ),
        "source_photo_root": str(source_photo_root),
        "output_root": str(output_root),
        "source_projects_opened_read_only": True,
        "summary": summary,
        "results": results,
    }
    with batch_report_path.open("x", encoding="utf-8") as output:
        json.dump(batch_report, output, indent=2, sort_keys=True)
        output.write("\n")
    return batch_report
