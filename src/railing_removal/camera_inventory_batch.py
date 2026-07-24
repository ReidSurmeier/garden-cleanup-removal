from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


InventoryProject = Callable[[Path], dict[str, Any]]
Progress = Callable[[str], None]


def _load_projects(path: Path) -> list[dict[str, Path | str]]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("project manifest schema_version must be 1")
    projects = value.get("projects")
    if not isinstance(projects, list) or not projects:
        raise ValueError("project manifest requires projects")
    scan_ids: set[str] = set()
    result: list[dict[str, Path | str]] = []
    for item in projects:
        scan_id = str(item.get("scan_id", "")).strip()
        if not scan_id or scan_id in scan_ids:
            raise ValueError("scan IDs must be nonempty and unique")
        scan_ids.add(scan_id)
        project = Path(item["project"]).resolve()
        if not project.is_file():
            raise FileNotFoundError(project)
        result.append({"scan_id": scan_id, "project": project})
    return result


def _validate_read_only(report: dict[str, Any]) -> None:
    if not report.get("project_opened_read_only"):
        raise ValueError("camera inventory lacks read-only provenance")


def run_camera_inventory_batch(
    manifest_path: Path,
    output_root: Path,
    *,
    inventory_project: InventoryProject,
    progress: Progress | None = None,
) -> dict[str, Any]:
    projects = _load_projects(manifest_path)
    output_root = output_root.resolve()
    batch_report_path = output_root / "batch-report.json"
    if batch_report_path.exists():
        raise FileExistsError(
            f"camera inventory batch is already finalized: "
            f"{batch_report_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda _: None)
    results: list[dict[str, Any]] = []
    for index, item in enumerate(projects, start=1):
        scan_id = str(item["scan_id"])
        project = Path(item["project"])
        destination = output_root / scan_id
        inventory_path = destination / "camera-inventory.json"
        if inventory_path.is_file():
            report = json.loads(
                inventory_path.read_text(encoding="utf-8")
            )
            _validate_read_only(report)
            progress(
                f"[{index}/{len(projects)}] already complete {scan_id}"
            )
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "complete",
                    "output": str(inventory_path),
                }
            )
            continue
        if destination.exists():
            progress(f"[{index}/{len(projects)}] partial output {scan_id}")
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "partial",
                    "output": str(destination),
                }
            )
            continue
        destination.mkdir(parents=True)
        progress(f"[{index}/{len(projects)}] processing {scan_id}")
        try:
            report = inventory_project(project)
            _validate_read_only(report)
            with inventory_path.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, sort_keys=True)
                output.write("\n")
        except Exception as error:
            progress(f"[{index}/{len(projects)}] failed {scan_id}: {error}")
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "output": str(destination),
                }
            )
            continue
        progress(f"[{index}/{len(projects)}] complete {scan_id}")
        results.append(
            {
                "scan_id": scan_id,
                "status": "complete",
                "output": str(inventory_path),
            }
        )
    summary = {
        status: sum(result["status"] == status for result in results)
        for status in ("complete", "partial", "failed")
    }
    batch_report = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "output_root": str(output_root),
        "source_projects_opened_read_only": True,
        "summary": summary,
        "results": results,
    }
    with batch_report_path.open("x", encoding="utf-8") as output:
        json.dump(batch_report, output, indent=2, sort_keys=True)
        output.write("\n")
    return batch_report
