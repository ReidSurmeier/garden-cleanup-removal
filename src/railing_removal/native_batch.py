from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from railing_removal.native_export import (
    canonicalize_native_cloud,
    export_native_cloud_readonly,
)


Progress = Callable[[str], None]
NativeExporter = Callable[..., dict[str, Any]]
Canonicalizer = Callable[..., dict[str, Any]]


def _load_projects(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    projects = value.get("projects")
    if value.get("schema_version") != 1 or not isinstance(projects, list):
        raise ValueError("invalid project manifest")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in projects:
        scan_id = str(item.get("scan_id", "")).strip()
        project = Path(item["project"]).resolve()
        if (
            not scan_id
            or scan_id in seen
            or Path(scan_id).name != scan_id
            or scan_id in {".", ".."}
        ):
            raise ValueError("unsafe or duplicate scan ID")
        if not project.is_file():
            raise FileNotFoundError(project)
        seen.add(scan_id)
        result.append({"scan_id": scan_id, "project": project})
    if not result:
        raise ValueError("project manifest is empty")
    return result


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def _final_report(
    manifest_path: Path,
    output_root: Path,
    results: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "output_root": str(output_root),
        "summary": {
            status: sum(
                result["status"] == status for result in results
            )
            for status in ("complete", "partial", "failed")
        },
        "results": results,
        **extra,
    }


def run_native_export_batch(
    manifest_path: Path,
    output_root: Path,
    metashape: Any,
    *,
    exporter: NativeExporter = export_native_cloud_readonly,
    progress: Progress | None = None,
) -> dict[str, Any]:
    projects = _load_projects(manifest_path)
    output_root = output_root.resolve()
    batch_report_path = output_root / "native-batch-report.json"
    if batch_report_path.exists():
        raise FileExistsError(
            f"native batch is already finalized: {batch_report_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda _: None)
    results: list[dict[str, Any]] = []

    for index, item in enumerate(projects, start=1):
        scan_id = item["scan_id"]
        destination = output_root / scan_id
        output = destination / "source-native.ply"
        report_path = destination / "native-export-report.json"
        if output.is_file() and report_path.is_file():
            progress(f"[{index}/{len(projects)}] already complete {scan_id}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "complete",
                    "output": str(output),
                    "source_point_count": report.get(
                        "source_point_count"
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
        destination.mkdir(parents=True)
        progress(f"[{index}/{len(projects)}] native export {scan_id}")
        try:
            report = exporter(item["project"], output, metashape)
            _write_json_new(report_path, report)
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
                "source_point_count": report.get("source_point_count"),
            }
        )
        progress(f"[{index}/{len(projects)}] complete {scan_id}")

    batch_report = _final_report(
        manifest_path,
        output_root,
        results,
        method="metashape-native-read-only",
    )
    _write_json_new(batch_report_path, batch_report)
    return batch_report


def run_canonicalize_native_batch(
    manifest_path: Path,
    native_root: Path,
    output_root: Path,
    *,
    stride: int = 1,
    canonicalizer: Canonicalizer = canonicalize_native_cloud,
    progress: Progress | None = None,
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be positive")
    projects = _load_projects(manifest_path)
    native_root = native_root.resolve()
    output_root = output_root.resolve()
    batch_report_path = output_root / "canonical-batch-report.json"
    if batch_report_path.exists():
        raise FileExistsError(
            f"canonical batch is already finalized: {batch_report_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda _: None)
    results: list[dict[str, Any]] = []

    for index, item in enumerate(projects, start=1):
        scan_id = item["scan_id"]
        native = native_root / scan_id / "source-native.ply"
        native_report_path = (
            native_root / scan_id / "native-export-report.json"
        )
        destination = output_root / scan_id
        output = destination / f"source-stride{stride}-zup.ply"
        report_path = destination / "export-report.json"
        if output.is_file() and report_path.is_file():
            progress(f"[{index}/{len(projects)}] already complete {scan_id}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
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
        if not native.is_file() or not native_report_path.is_file():
            progress(f"[{index}/{len(projects)}] missing native {scan_id}")
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "failed",
                    "error": "native export is incomplete",
                    "output": str(destination),
                }
            )
            continue

        native_report = json.loads(
            native_report_path.read_text(encoding="utf-8")
        )
        destination.mkdir(parents=True)
        progress(f"[{index}/{len(projects)}] canonicalize {scan_id}")
        try:
            report = canonicalizer(
                native,
                output,
                native_report["coordinate_frame"],
                stride=stride,
            )
            _write_json_new(report_path, report)
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

    batch_report = _final_report(
        manifest_path,
        output_root,
        results,
        method="vectorized-native-to-zup",
        stride=stride,
        native_root=str(native_root),
    )
    _write_json_new(batch_report_path, batch_report)
    return batch_report
