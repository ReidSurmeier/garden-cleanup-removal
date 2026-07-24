from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from plant_cleanup.source_photo_semantic import (
    aggregate_source_photo_votes,
)


Progress = Callable[[str], None]
Aggregate = Callable[..., dict[str, Any]]


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
        photo_dir = Path(item["photo_dir"]).resolve()
        if not photo_dir.is_dir():
            raise FileNotFoundError(photo_dir)
        result.append({"scan_id": scan_id, "photo_dir": photo_dir})
    return result


def _validate_report(report: dict[str, Any]) -> None:
    if not report.get("project_opened_read_only"):
        raise ValueError("source-photo report lacks read-only provenance")


def run_source_photo_semantic_batch(
    manifest_path: Path,
    canonical_root: Path,
    native_root: Path,
    inventory_root: Path,
    output_root: Path,
    *,
    predictor: object,
    model_id: str,
    plant_labels: tuple[str, ...],
    stride: int = 8,
    camera_count: int = 12,
    maximum_dimension: int = 768,
    aggregate: Aggregate = aggregate_source_photo_votes,
    progress: Progress | None = None,
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be positive")
    projects = _load_projects(manifest_path)
    canonical_root = canonical_root.resolve()
    native_root = native_root.resolve()
    inventory_root = inventory_root.resolve()
    output_root = output_root.resolve()
    batch_report_path = output_root / "batch-report.json"
    if batch_report_path.exists():
        raise FileExistsError(
            f"source-photo batch is already finalized: {batch_report_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda _: None)
    results: list[dict[str, Any]] = []
    for index, item in enumerate(projects, start=1):
        scan_id = str(item["scan_id"])
        destination = output_root / scan_id
        report_path = destination / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _validate_report(report)
            progress(
                f"[{index}/{len(projects)}] already complete {scan_id}"
            )
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "complete",
                    "point_count": int(report["point_count"]),
                    "unseen_point_count": int(
                        report["unseen_point_count"]
                    ),
                    "output": str(destination),
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
        canonical_cloud = (
            canonical_root
            / scan_id
            / f"source-stride{stride}-zup.ply"
        )
        native_cloud = native_root / scan_id / "source-native.ply"
        inventory = (
            inventory_root / scan_id / "camera-inventory.json"
        )
        photo_dir = Path(item["photo_dir"])
        progress(f"[{index}/{len(projects)}] processing {scan_id}")
        try:
            for dependency in (
                canonical_cloud,
                native_cloud,
                inventory,
            ):
                if not dependency.is_file():
                    raise FileNotFoundError(dependency)
            report = aggregate(
                canonical_cloud.resolve(),
                native_cloud.resolve(),
                inventory.resolve(),
                photo_dir,
                destination,
                predictor=predictor,
                model_id=model_id,
                plant_labels=plant_labels,
                camera_count=camera_count,
                maximum_dimension=maximum_dimension,
            )
            _validate_report(report)
            if not report_path.is_file():
                raise FileNotFoundError(report_path)
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
                "point_count": int(report["point_count"]),
                "unseen_point_count": int(
                    report["unseen_point_count"]
                ),
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
        "canonical_root": str(canonical_root),
        "native_root": str(native_root),
        "inventory_root": str(inventory_root),
        "output_root": str(output_root),
        "model": model_id,
        "plant_labels": list(plant_labels),
        "camera_count": camera_count,
        "maximum_dimension": maximum_dimension,
        "source_projects_opened_read_only": True,
        "summary": summary,
        "results": results,
    }
    with batch_report_path.open("x", encoding="utf-8") as output:
        json.dump(batch_report, output, indent=2, sort_keys=True)
        output.write("\n")
    return batch_report
