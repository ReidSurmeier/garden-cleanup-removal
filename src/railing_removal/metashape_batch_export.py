from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from railing_removal.metashape_reader import export_reader_cloud_readonly


Progress = Callable[[str], None]
Exporter = Callable[..., dict[str, Any]]


def load_project_manifest(path: Path) -> list[dict[str, Any]]:
    manifest_path = path.resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("project manifest schema_version must be 1")
    projects = value.get("projects")
    if not isinstance(projects, list) or not projects:
        raise ValueError("project manifest requires a nonempty projects list")

    scan_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in projects:
        scan_id = str(item.get("scan_id", "")).strip()
        if (
            not scan_id
            or scan_id in scan_ids
            or Path(scan_id).name != scan_id
            or scan_id in {".", ".."}
        ):
            raise ValueError(
                "project scan IDs must be unique safe directory names"
            )
        scan_ids.add(scan_id)
        project = Path(item["project"]).resolve()
        if not project.is_file():
            raise FileNotFoundError(project)
        result.append({"scan_id": scan_id, "project": project})
    return result


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def run_metashape_export_batch(
    manifest_path: Path,
    output_root: Path,
    metashape: Any,
    *,
    stride: int = 1,
    exporter: Exporter = export_reader_cloud_readonly,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Export many projects read-only in one Metashape Python process."""

    if stride < 1:
        raise ValueError("stride must be positive")
    projects = load_project_manifest(manifest_path)
    output_root = output_root.resolve()
    batch_report_path = output_root / "batch-export-report.json"
    if batch_report_path.exists():
        raise FileExistsError(
            f"batch export is already finalized: {batch_report_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda _: None)

    results: list[dict[str, Any]] = []
    output_name = f"source-stride{stride}-zup.ply"
    for index, item in enumerate(projects, start=1):
        scan_id = item["scan_id"]
        destination = output_root / scan_id
        output = destination / output_name
        export_report = destination / "export-report.json"
        if output.is_file() and export_report.is_file():
            progress(f"[{index}/{len(projects)}] already complete {scan_id}")
            report = json.loads(export_report.read_text(encoding="utf-8"))
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "complete",
                    "output": str(output),
                    "exported_point_count": report.get(
                        "exported_point_count"
                    ),
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

        progress(f"[{index}/{len(projects)}] exporting {scan_id}")
        destination.mkdir(parents=True)
        try:
            report = exporter(
                item["project"],
                output,
                metashape,
                stride=stride,
            )
            _write_json_new(export_report, report)
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
        results.append(
            {
                "scan_id": scan_id,
                "status": "complete",
                "output": str(output),
                "exported_point_count": report.get(
                    "exported_point_count"
                ),
            }
        )
        progress(f"[{index}/{len(projects)}] complete {scan_id}")

    summary = {
        status: sum(result["status"] == status for result in results)
        for status in ("complete", "partial", "failed")
    }
    batch_report = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "output_root": str(output_root),
        "stride": stride,
        "summary": summary,
        "results": results,
    }
    _write_json_new(batch_report_path, batch_report)
    return batch_report
