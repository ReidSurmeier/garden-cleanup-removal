from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from plant_cleanup.adaptive_profile import build_adaptive_config
from railing_removal.native_batch import _load_projects


ConfigBuilder = Callable[[Path, dict[str, Any]], dict[str, Any]]
Progress = Callable[[str], None]


def _write_json_atomic_new(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(f"{path.name}.partial-{uuid4().hex}")
    with partial.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")
    partial.rename(path)


def _load_scene_catalog(
    path: Path,
    scan_ids: set[str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    classes = value.get("classes")
    assignments = value.get("assignments")
    plant_prompt = str(value.get("plant_prompt", "")).strip()
    if (
        value.get("schema_version") != 1
        or not isinstance(classes, dict)
        or not isinstance(assignments, dict)
        or not plant_prompt
    ):
        raise ValueError("invalid scene plan catalog")
    unknown_scans = set(assignments) - scan_ids
    if unknown_scans:
        raise ValueError(
            f"scene catalog contains unknown scan: {sorted(unknown_scans)[0]}"
        )
    normalized: dict[str, list[str]] = {}
    for scan_id, assigned in assignments.items():
        if (
            not isinstance(assigned, list)
            or not assigned
            or len(set(assigned)) != len(assigned)
        ):
            raise ValueError(f"invalid class assignment for {scan_id}")
        unknown_classes = set(assigned) - set(classes)
        if unknown_classes:
            raise ValueError(
                f"scene catalog contains unknown class: "
                f"{sorted(unknown_classes)[0]}"
            )
        normalized[scan_id] = [str(class_id) for class_id in assigned]
    return value, normalized


def _scene_plan(
    catalog: dict[str, Any],
    scan_id: str,
    class_ids: list[str],
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for class_id in class_ids:
        object_class = copy.deepcopy(catalog["classes"][class_id])
        object_class["id"] = class_id
        values.append(object_class)
    return {
        "schema_version": 1,
        "scan_id": scan_id,
        "target_intent": catalog.get(
            "target_intent",
            "Keep coherent plants including roots, stems, trunks, branches, "
            "and leaves. Remove only the assigned non-plant classes.",
        ),
        "plant_prompt": catalog["plant_prompt"],
        "classes": values,
    }


def build_adaptive_batch(
    project_manifest_path: Path,
    canonical_root: Path,
    output_root: Path,
    base_config_path: Path,
    *,
    stride: int = 8,
    scene_catalog_path: Path | None = None,
    config_builder: ConfigBuilder = build_adaptive_config,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build resumable source-bound configs and one cleanup manifest."""

    if stride < 1:
        raise ValueError("stride must be positive")
    projects = _load_projects(project_manifest_path)
    canonical_root = canonical_root.resolve()
    output_root = output_root.resolve()
    base_config_path = base_config_path.resolve()
    manifest_path = output_root / "cleanup-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"adaptive batch is already finalized: {manifest_path}"
        )
    if not base_config_path.is_file():
        raise FileNotFoundError(base_config_path)
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    catalog: dict[str, Any] | None = None
    assignments: dict[str, list[str]] = {}
    if scene_catalog_path is not None:
        scene_catalog_path = scene_catalog_path.resolve()
        if not scene_catalog_path.is_file():
            raise FileNotFoundError(scene_catalog_path)
        catalog, assignments = _load_scene_catalog(
            scene_catalog_path,
            {item["scan_id"] for item in projects},
        )
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    plan_root = output_root / "scene-plans"
    progress = progress or (lambda _: None)
    scans: list[dict[str, Any]] = []

    for index, item in enumerate(projects, start=1):
        scan_id = item["scan_id"]
        source = (
            canonical_root
            / scan_id
            / f"source-stride{stride}-zup.ply"
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"canonical source is incomplete: {source}"
            )
        config_path = config_root / f"{scan_id}.json"
        if config_path.exists():
            progress(f"[{index}/{len(projects)}] config exists {scan_id}")
            value = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"invalid existing config: {config_path}")
        else:
            progress(f"[{index}/{len(projects)}] profile {scan_id}")
            config = config_builder(source, base_config)
            _write_json_atomic_new(config_path, config)
        item_manifest = {
            "scan_id": scan_id,
            "source": str(source.resolve()),
            "config": str(config_path.resolve()),
        }
        if scan_id in assignments:
            plan_root.mkdir(parents=True, exist_ok=True)
            plan_path = plan_root / f"{scan_id}.json"
            if plan_path.exists():
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                if plan.get("scan_id") != scan_id:
                    raise ValueError(
                        f"existing plan does not match scan: {plan_path}"
                    )
            else:
                assert catalog is not None
                _write_json_atomic_new(
                    plan_path,
                    _scene_plan(catalog, scan_id, assignments[scan_id]),
                )
            item_manifest["scene_plan"] = str(plan_path.resolve())
        scans.append(item_manifest)

    manifest = {
        "schema_version": 1,
        "project_manifest": str(project_manifest_path.resolve()),
        "canonical_root": str(canonical_root),
        "base_config": str(base_config_path),
        "scene_catalog": (
            str(scene_catalog_path)
            if scene_catalog_path is not None
            else None
        ),
        "stride": stride,
        "scans": scans,
    }
    _write_json_atomic_new(manifest_path, manifest)
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "summary": {"complete": len(scans)},
    }
