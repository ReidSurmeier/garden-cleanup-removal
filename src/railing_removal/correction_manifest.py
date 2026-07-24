from __future__ import annotations

import copy
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
    """Select unfinished scans while preserving assigned object evidence."""
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

    corrections: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict) or result.get("status") == "complete":
            continue
        scan_id = str(result.get("scan_id", "")).strip()
        source = by_id.get(scan_id)
        if source is None:
            raise ValueError(
                f"baseline report references unknown scan: {scan_id}"
            )
        correction = {
            "scan_id": scan_id,
            "source": str(source["source"]),
            "config": str(source["config"]),
            "review_artifacts": False,
        }
        if source.get("scene_plan") is not None:
            correction["scene_plan"] = str(source["scene_plan"])
        corrections.append(correction)
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


def build_targeted_correction_manifest(
    source_manifest_path: Path,
    scene_catalog_path: Path,
    assignment_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build additive reruns selected by visual QA and named object classes."""
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

    catalog = _read_object(scene_catalog_path)
    classes = catalog.get("classes")
    plant_prompt = str(catalog.get("plant_prompt", "")).strip()
    if not isinstance(classes, dict) or not plant_prompt:
        raise ValueError("scene catalog requires classes and plant_prompt")

    assignment_document = _read_object(assignment_path)
    assignments = assignment_document.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("targeted correction assignments must not be empty")
    class_overrides = assignment_document.get("class_overrides", {})
    if not isinstance(class_overrides, dict):
        raise ValueError("targeted correction class_overrides must be an object")
    unknown_override_scans = set(class_overrides) - set(assignments)
    if unknown_override_scans:
        raise ValueError(
            "class override references unassigned scan: "
            f"{sorted(unknown_override_scans)[0]}"
        )

    plan_root = output_path.parent / f"{output_path.stem}-scene-plans"
    corrections: list[dict[str, Any]] = []
    plans: list[tuple[Path, dict[str, Any]]] = []
    for scan_id_value, class_ids in assignments.items():
        scan_id = str(scan_id_value).strip()
        if not scan_id or Path(scan_id).name != scan_id:
            raise ValueError(f"invalid targeted scan id: {scan_id!r}")
        source = by_id.get(scan_id)
        if source is None:
            raise ValueError(f"targeted correction references unknown scan: {scan_id}")
        if (
            not isinstance(class_ids, list)
            or not class_ids
            or len(set(class_ids)) != len(class_ids)
        ):
            raise ValueError(f"invalid class assignment for {scan_id}")
        unknown_classes = set(class_ids) - set(classes)
        if unknown_classes:
            raise ValueError(
                "targeted correction contains unknown class: "
                f"{sorted(unknown_classes)[0]}"
            )
        scan_overrides = class_overrides.get(scan_id, {})
        if not isinstance(scan_overrides, dict):
            raise ValueError(f"invalid class overrides for {scan_id}")
        unknown_override_classes = set(scan_overrides) - set(class_ids)
        if unknown_override_classes:
            raise ValueError(
                "class override references unassigned class: "
                f"{sorted(unknown_override_classes)[0]}"
            )
        plan_path = plan_root / f"{scan_id}.json"
        if plan_path.exists():
            raise FileExistsError(plan_path)
        object_classes: list[dict[str, Any]] = []
        for class_id in class_ids:
            object_class = copy.deepcopy(classes[class_id])
            override = scan_overrides.get(class_id, {})
            if not isinstance(override, dict):
                raise ValueError(
                    f"invalid {class_id} override for {scan_id}"
                )
            unknown_override_keys = set(override) - {
                "manual_seed_regions",
                "line_completion_parameters",
            }
            if unknown_override_keys:
                raise ValueError(
                    "unsupported class override: "
                    f"{sorted(unknown_override_keys)[0]}"
                )
            if "line_completion_parameters" in override and not isinstance(
                override["line_completion_parameters"],
                dict,
            ):
                raise ValueError(
                    "line_completion_parameters must be an object"
                )
            object_class.update(copy.deepcopy(override))
            object_class["id"] = class_id
            object_classes.append(object_class)
        plans.append(
            (
                plan_path,
                {
                    "schema_version": 1,
                    "scan_id": scan_id,
                    "target_intent": str(
                        catalog.get(
                            "target_intent",
                            "Keep coherent plants and remove only assigned "
                            "non-plant objects.",
                        )
                    ),
                    "plant_prompt": plant_prompt,
                    "classes": object_classes,
                },
            )
        )
        corrections.append(
            {
                "scan_id": scan_id,
                "source": str(source["source"]),
                "config": str(source["config"]),
                "scene_plan": str(plan_path.resolve()),
                "review_artifacts": False,
            }
        )

    plan_root.mkdir(parents=True, exist_ok=False)
    for plan_path, plan in plans:
        with plan_path.open("x", encoding="utf-8") as output:
            json.dump(plan, output, indent=2, sort_keys=True)
            output.write("\n")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_manifest": str(source_manifest_path.resolve()),
        "scene_catalog": str(scene_catalog_path.resolve()),
        "assignment_document": str(assignment_path.resolve()),
        "scans": corrections,
    }
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return manifest
