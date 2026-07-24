from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4


ARTIFACT_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _write_json_atomic_new(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(f"{path.name}.partial-{uuid4().hex}")
    with partial.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")
    partial.rename(path)


def _copy_atomic_new(source: Path, destination: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    partial = destination.with_name(
        f"{destination.name}.partial-{uuid4().hex}"
    )
    size = 0
    with source.open("rb") as reader, partial.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)
            digest.update(block)
            size += len(block)
    partial.rename(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _load_project_roots(path: Path) -> dict[str, Path]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    projects = value.get("projects")
    source_root_value = value.get("source_root")
    if (
        value.get("schema_version") != 1
        or not isinstance(projects, list)
        or not source_root_value
    ):
        raise ValueError("invalid project manifest")
    source_root = Path(source_root_value).resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    result: dict[str, Path] = {}
    for item in projects:
        scan_id = str(item.get("scan_id", "")).strip()
        project = Path(item["project"]).resolve()
        if (
            not scan_id
            or scan_id in result
            or Path(scan_id).name != scan_id
            or not project.is_file()
        ):
            raise ValueError("unsafe or duplicate project manifest entry")
        try:
            project.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"project lies outside explicit source root: {project}"
            ) from error
        result[scan_id] = project.parent
    return result


def _load_batch_results(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    results: dict[str, dict[str, Any]] = {}
    for item in value.get("results", []):
        scan_id = str(item.get("scan_id", "")).strip()
        if not scan_id or scan_id in results:
            raise ValueError("unsafe or duplicate batch result")
        results[scan_id] = item
    return results


def publish_outputs(
    project_manifest_path: Path,
    batch_report_path: Path,
    publication_plan_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Add only explicitly planned, versioned derived files to scan folders."""

    report_path = report_path.resolve()
    if report_path.exists():
        raise FileExistsError(
            f"refusing to overwrite publication report: {report_path}"
        )
    project_roots = _load_project_roots(project_manifest_path)
    batch_results = _load_batch_results(batch_report_path)
    plan = json.loads(
        publication_plan_path.resolve().read_text(encoding="utf-8")
    )
    scans = plan.get("scans")
    artifact_tag = str(plan.get("artifact_tag", "")).strip()
    if (
        plan.get("schema_version") != 1
        or not isinstance(scans, list)
        or not scans
        or ARTIFACT_TAG.fullmatch(artifact_tag) is None
    ):
        raise ValueError("invalid publication plan")

    planned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in scans:
        scan_id = str(item.get("scan_id", "")).strip()
        action = str(item.get("action", "")).strip()
        if (
            not scan_id
            or scan_id in seen
            or scan_id not in project_roots
            or scan_id not in batch_results
            or action not in {"publish_clean", "publish_flag"}
        ):
            raise ValueError("unsafe or invalid publication plan entry")
        seen.add(scan_id)
        batch_item = batch_results[scan_id]
        if batch_item["status"] != "complete":
            raise ValueError(f"cannot publish incomplete scan: {scan_id}")
        scan_root = project_roots[scan_id]
        cleanup_root = Path(batch_item["output"]).resolve()
        if action == "publish_clean":
            sources = {
                f"plant-cleaned-{artifact_tag}.ply": (
                    cleanup_root
                    / "final"
                    / "plant-cleaned-color-corrected.ply"
                ),
                f"plant-cleanup-before-{artifact_tag}.png": (
                    cleanup_root
                    / "final"
                    / "render-source"
                    / "front-rgb.png"
                ),
                f"plant-cleanup-after-{artifact_tag}.png": (
                    cleanup_root
                    / "final"
                    / "render-plant"
                    / "front-rgb.png"
                ),
                f"plant-cleanup-report-{artifact_tag}.json": (
                    cleanup_root / "run-report.json"
                ),
            }
            for source in sources.values():
                if not source.is_file():
                    raise FileNotFoundError(source)
            destinations = {
                name: scan_root / name for name in sources
            }
        else:
            reason = str(item.get("reason", "")).strip()
            if not reason:
                raise ValueError(f"flag requires a reason: {scan_id}")
            sources = {}
            destinations = {
                f"plant-cleanup-flag-{artifact_tag}.json": (
                    scan_root
                    / f"plant-cleanup-flag-{artifact_tag}.json"
                )
            }
        for destination in destinations.values():
            if destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite publication: {destination}"
                )
        planned.append(
            {
                "scan_id": scan_id,
                "action": action,
                "reason": str(item.get("reason", "")).strip() or None,
                "sources": sources,
                "destinations": destinations,
            }
        )

    results: list[dict[str, Any]] = []
    for item in planned:
        artifacts: list[dict[str, Any]] = []
        if item["action"] == "publish_clean":
            for name, source in item["sources"].items():
                artifacts.append(
                    _copy_atomic_new(source, item["destinations"][name])
                )
        else:
            destination = next(iter(item["destinations"].values()))
            _write_json_atomic_new(
                destination,
                {
                    "schema_version": 1,
                    "scan_id": item["scan_id"],
                    "status": "no_coherent_plant",
                    "reason": item["reason"],
                    "artifact_tag": artifact_tag,
                    "batch_report": str(batch_report_path.resolve()),
                },
            )
            artifacts.append(
                {
                    "destination": str(destination),
                    "bytes": destination.stat().st_size,
                }
            )
        results.append(
            {
                "scan_id": item["scan_id"],
                "action": item["action"],
                "artifacts": artifacts,
            }
        )

    report = {
        "schema_version": 1,
        "project_manifest": str(project_manifest_path.resolve()),
        "batch_report": str(batch_report_path.resolve()),
        "publication_plan": str(publication_plan_path.resolve()),
        "artifact_tag": artifact_tag,
        "summary": {
            action: sum(result["action"] == action for result in results)
            for action in ("publish_clean", "publish_flag")
        },
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic_new(report_path, report)
    return report
