from __future__ import annotations

import hashlib
import json
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

import numpy as np

from plant_cleanup.classification import (
    ClassificationParameters,
    classify_points,
)
from plant_cleanup.clipseg_votes import (
    HuggingFaceClipSegPredictor,
    aggregate_clipseg_votes,
)
from plant_cleanup.cloud_render import render_cloud_views
from plant_cleanup.color_correct import ColorParameters, correct_cloud_colors
from plant_cleanup.dense_semantic import (
    DensePropagationParameters,
    HuggingFaceOneFormerPredictor,
    aggregate_dense_semantic_votes,
    propagate_dense_semantic_evidence,
)
from plant_cleanup.geometry_cleanup import (
    CleanupParameters,
    _support_height,
    write_decision_cloud,
)
from plant_cleanup.plyio import read_cloud
from plant_cleanup.root_safe import (
    geometry_decisions,
    validate_config,
)
from plant_cleanup.sam2_votes import (
    HuggingFaceSam2Predictor,
    aggregate_sam2_votes,
)
from plant_cleanup.scene_evidence import run_scene_evidence
from plant_cleanup.semantic_refine import SemanticParameters, refine_with_semantics
from plant_cleanup.web_preview import export_web_preview
from railing_removal.completion import complete_railing_lines
from railing_removal.floor import remove_uncertain_floor


Progress = Callable[[str], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_viewer(
    *,
    source: Path,
    previous: Path,
    plant: Path,
    conservative: Path,
    rejected: Path,
    uncertain: Path,
    output: Path,
) -> dict[str, Any]:
    viewer_data = output / "viewer-data"
    layers = {
        "source": source,
        "previous": previous,
        "plant": plant,
        "conservative": conservative,
        "rejected": rejected,
        "uncertain": uncertain,
    }
    reports: dict[str, dict[str, Any]] = {}
    for name, path in layers.items():
        report = export_web_preview(path, viewer_data / f"{name}.bin")
        report["url"] = f"viewer-data/{name}.bin"
        reports[name] = report
    nonempty_reports = [
        report
        for report in reports.values()
        if report["bounds"] is not None
    ]
    if not nonempty_reports:
        raise ValueError("viewer requires at least one nonempty layer")
    combined_min = [
        min(report["bounds"]["min"][axis] for report in nonempty_reports)
        for axis in range(3)
    ]
    combined_max = [
        max(report["bounds"]["max"][axis] for report in nonempty_reports)
        for axis in range(3)
    ]
    manifest = {
        "format": "xyz-float32-rgba-uint8-interleaved",
        "record_bytes": 16,
        "bounds": {"min": combined_min, "max": combined_max},
        "layers": reports,
    }
    _write_json(viewer_data / "manifest.json", manifest)
    template = files("railing_removal").joinpath("assets/viewer.html")
    with template.open("rb") as source_template:
        viewer = output / "viewer.html"
        viewer.parent.mkdir(parents=True, exist_ok=True)
        with viewer.open("wb") as destination:
            shutil.copyfileobj(source_template, destination)
    shutil.copyfile(viewer, output / "index.html")
    return manifest


def run_full_cleanup(
    source_path: Path,
    output_dir: Path,
    *,
    config_path: Path,
    railing_plan_path: Path | None = None,
    clipseg_predictor: Any | None = None,
    sam2_predictor: Any | None = None,
    dense_predictor: Any | None = None,
    build_review_artifacts: bool = True,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the complete approved geometry, vision, floor, and railing pipeline."""

    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    railing_plan_path = (
        railing_plan_path.resolve()
        if railing_plan_path is not None
        else None
    )
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    progress = progress or (lambda _: None)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = (
        json.loads(railing_plan_path.read_text(encoding="utf-8"))
        if railing_plan_path is not None
        else None
    )
    validate_config(config)
    cloud = read_cloud(source_path)
    source_hash = _sha256(source_path)
    profile = config.get("profile")
    if profile and (
        profile.get("source_sha256") != source_hash
        or int(profile.get("source_point_count", -1)) != len(cloud)
    ):
        raise ValueError("config profile does not match the source cloud")
    output_dir.mkdir(parents=True)

    coordinates = np.column_stack((cloud["x"], cloud["y"], cloud["z"]))
    normals = np.column_stack((cloud["nx"], cloud["ny"], cloud["nz"]))
    rgb = np.column_stack((cloud["red"], cloud["green"], cloud["blue"]))

    progress("geometry")
    target = classify_points(
        coordinates,
        rgb,
        ClassificationParameters(**config["target_classification"]),
    )
    decisions = geometry_decisions(target.reasons)
    target_dir = output_dir / "target"
    target_dir.mkdir()
    np.save(target_dir / "decision-codes.npy", decisions)
    _write_json(target_dir / "classification-report.json", target.report)

    support_values = config["support_estimation"]
    support_parameters = CleanupParameters(
        support_bin_size=support_values["support_bin_size"],
        support_clearance=support_values["support_clearance"],
        support_normal_min=support_values["support_normal_min"],
        vegetation_excess_green_min=support_values[
            "vegetation_excess_green_min"
        ],
    )
    support_height = (
        float(support_values["support_height"])
        if "support_height" in support_values
        else _support_height(cloud, support_parameters)
    )
    support_cutoff = support_height + support_parameters.support_clearance

    vision = config["vision"]
    progress("render-eight-views")
    render_dir = output_dir / "render-source-orbit"
    render_report = render_cloud_views(
        source_path,
        render_dir,
        size=vision["render_size"],
        point_radius=vision["render_point_radius"],
        yaw_degrees=tuple(vision["orbit_degrees"]),
    )
    _write_json(render_dir / "render-report.json", render_report)

    progress("load-models")
    clipseg_predictor = clipseg_predictor or HuggingFaceClipSegPredictor(
        vision["clipseg_model"]
    )
    sam2_predictor = sam2_predictor or HuggingFaceSam2Predictor(
        vision["sam2_model"]
    )

    progress("generic-semantic-evidence")
    clipseg_dir = output_dir / "vision-clipseg"
    aggregate_clipseg_votes(
        source_path,
        render_dir,
        clipseg_dir,
        predictor=clipseg_predictor,
        model_id=vision["clipseg_model"],
        prompts=tuple(vision["prompts"]),
        confidence_min=vision["clipseg_confidence_min"],
        margin_min=vision["clipseg_margin_min"],
    )
    sam2_dir = output_dir / "vision-sam2"
    aggregate_sam2_votes(
        source_path,
        render_dir,
        clipseg_dir,
        sam2_dir,
        predictor=sam2_predictor,
        model_id=vision["sam2_model"],
        prompt_grid=tuple(vision["prompt_grid"]),
        margin_min=vision["prompt_margin_min"],
    )

    generic_plant_votes = np.load(sam2_dir / "plant-votes.npy")
    generic_background_votes = np.load(sam2_dir / "planter-votes.npy")
    plan_classes = {
        str(object_class["id"]): object_class
        for object_class in (plan or {}).get("classes", [])
    }
    non_railing_classes = {
        class_id: object_class
        for class_id, object_class in plan_classes.items()
        if class_id != "railing"
    }
    scene_evidence: Path | None = None
    scene_report: dict[str, Any] = {
        "status": "skipped",
        "reason": "no scene plan supplied",
    }
    if non_railing_classes:
        progress("non-plant-semantic-evidence")
        scene_values = config.get("scene_evidence", {})
        scene_evidence = output_dir / "scene-evidence"
        scene_report = run_scene_evidence(
            source_path,
            render_dir,
            scene_evidence,
            plan,
            clipseg_predictor=clipseg_predictor,
            sam2_predictor=sam2_predictor,
            clipseg_model=scene_values.get(
                "clipseg_model",
                vision["clipseg_model"],
            ),
            sam2_model=scene_values.get(
                "sam2_model",
                vision["sam2_model"],
            ),
            confidence_min=float(
                scene_values.get("clipseg_confidence_min", 0.2)
            ),
            clipseg_margin_min=float(
                scene_values.get("clipseg_margin_min", 0.01)
            ),
            sam2_margin_min=float(
                scene_values.get("sam2_margin_min", 0.05)
            ),
            prompt_grid=tuple(
                scene_values.get("prompt_grid", [3, 6])
            ),
        )

    progress("semantic-refinement")
    semantic_dir = output_dir / "semantic"
    fused_scene = (
        scene_evidence / "fused"
        if scene_evidence is not None
        else None
    )
    semantic_report = refine_with_semantics(
        source_path,
        decisions,
        generic_plant_votes,
        generic_background_votes,
        support_cutoff=support_cutoff,
        output_dir=semantic_dir,
        parameters=SemanticParameters(**config["semantic_refinement"]),
        background_class_votes=(
            {
                class_id: np.load(
                    fused_scene / f"{class_id}-votes.npy"
                )
                for class_id in non_railing_classes
            }
            if fused_scene is not None
            else None
        ),
        background_class_plant_votes=(
            {
                class_id: np.load(
                    fused_scene / f"{class_id}-plant-votes.npy"
                )
                for class_id in non_railing_classes
            }
            if fused_scene is not None
            else None
        ),
        background_class_policies={
            class_id: str(
                object_class.get(
                    "decision_policy",
                    "strict_background",
                )
            )
            for class_id, object_class in non_railing_classes.items()
        },
    )
    _write_json(semantic_dir / "semantic-report.json", semantic_report)

    progress("uncertain-floor-removal")
    semantic_decisions = np.load(semantic_dir / "decision-codes.npy")
    floor_keep, floor_report = remove_uncertain_floor(
        coordinates,
        normals=normals,
        rgb=rgb,
        decisions=semantic_decisions,
        plant_votes=generic_plant_votes,
        background_votes=generic_background_votes,
    )
    floor_dir = output_dir / "floor"
    floor_dir.mkdir()
    floor_plant = floor_dir / "plant-floor-corrected.ply"
    floor_rejected = floor_dir / "rejected-floor-corrected.ply"
    write_decision_cloud(cloud, floor_plant, floor_keep, semantic_decisions)
    write_decision_cloud(cloud, floor_rejected, ~floor_keep, semantic_decisions)
    _write_json(floor_dir / "floor-report.json", floor_report)

    dense_values = config.get("dense_semantic")
    if dense_values:
        progress("dense-semantic-evidence")
        dense_predictor = (
            dense_predictor
            or HuggingFaceOneFormerPredictor(dense_values["model"])
        )
        dense_dir = output_dir / "dense-semantic"
        dense_report = aggregate_dense_semantic_votes(
            source_path,
            render_dir,
            dense_dir,
            predictor=dense_predictor,
            model_id=dense_values["model"],
            plant_labels=tuple(dense_values["plant_labels"]),
            background_labels=tuple(
                dense_values["background_labels"]
            ),
        )
        strict_keep, conservative_keep, propagation_report = (
            propagate_dense_semantic_evidence(
                coordinates,
                normals=normals,
                candidate_mask=floor_keep,
                plant_votes=np.load(dense_dir / "plant-votes.npy"),
                background_votes=np.load(
                    dense_dir / "background-votes.npy"
                ),
                support_plane_coefficients=tuple(
                    semantic_report["support_plane"]["coefficients"]
                ),
                vertical_span=float(config["profile"]["vertical_span"]),
                parameters=DensePropagationParameters(
                    **dense_values["propagation"]
                ),
            )
        )
        _write_json(
            dense_dir / "propagation-report.json",
            propagation_report,
        )
    else:
        strict_keep = floor_keep.copy()
        conservative_keep = floor_keep.copy()
        dense_report = {
            "status": "skipped",
            "reason": "dense_semantic is not configured",
        }
        propagation_report = dense_report

    if "railing" in plan_classes:
        progress("railing-semantic-evidence")
        if scene_evidence is None:
            scene_values = config.get("scene_evidence", {})
            scene_evidence = output_dir / "scene-evidence"
            scene_report = run_scene_evidence(
                source_path,
                render_dir,
                scene_evidence,
                plan,
                clipseg_predictor=clipseg_predictor,
                sam2_predictor=sam2_predictor,
                clipseg_model=scene_values.get(
                    "clipseg_model",
                    vision["clipseg_model"],
                ),
                sam2_model=scene_values.get(
                    "sam2_model",
                    vision["sam2_model"],
                ),
                confidence_min=float(
                    scene_values.get("clipseg_confidence_min", 0.2)
                ),
                clipseg_margin_min=float(
                    scene_values.get("clipseg_margin_min", 0.01)
                ),
                sam2_margin_min=float(
                    scene_values.get("sam2_margin_min", 0.05)
                ),
                prompt_grid=tuple(
                    scene_values.get("prompt_grid", [3, 6])
                ),
            )
        progress("railing-line-completion")
        fused = scene_evidence / "fused"
        railing_reject, completion_report = complete_railing_lines(
            coordinates,
            rgb=rgb,
            candidate_mask=floor_keep,
            railing_votes=np.load(fused / "railing-votes.npy"),
            plant_votes=np.load(fused / "railing-plant-votes.npy"),
        )
    else:
        progress("railing-stage-skipped")
        railing_reject = np.zeros(len(cloud), dtype=bool)
        completion_report = {
            "schema_version": 1,
            "status": "skipped",
            "candidate_point_count": int(floor_keep.sum()),
            "completed_point_count": 0,
            "reason": (
                "no scene plan supplied"
                if plan is None
                else "scene plan contains no railing class"
            ),
        }
    final_keep = strict_keep & ~railing_reject
    final_conservative_keep = conservative_keep & ~railing_reject
    final_dir = output_dir / "final"
    final_dir.mkdir()
    plant_path = final_dir / "plant-cleaned.ply"
    conservative_path = final_dir / "plant-cleaned-conservative.ply"
    rejected_path = final_dir / "rejected-cleaned.ply"
    write_decision_cloud(cloud, plant_path, final_keep, semantic_decisions)
    write_decision_cloud(
        cloud,
        conservative_path,
        final_conservative_keep,
        semantic_decisions,
    )
    write_decision_cloud(cloud, rejected_path, ~final_keep, semantic_decisions)
    _write_json(final_dir / "railing-completion-report.json", completion_report)

    progress("color-correction")
    color_path = final_dir / "plant-cleaned-color-corrected.ply"
    color_report = correct_cloud_colors(
        plant_path,
        color_path,
        ColorParameters(**config["color_correction"]),
    )
    _write_json(final_dir / "color-report.json", color_report)

    if build_review_artifacts:
        progress("proof-renders")
        for name, path in (
            ("source", source_path),
            ("plant", plant_path),
            ("conservative", conservative_path),
            ("rejected", rejected_path),
        ):
            proof = render_cloud_views(
                path,
                final_dir / f"render-{name}",
                size=config["proof_render_size"],
                point_radius=1,
            )
            _write_json(
                final_dir / f"render-{name}" / "render-report.json",
                proof,
            )

        progress("web-viewer")
        viewer_manifest = _build_viewer(
            source=source_path,
            previous=floor_plant,
            plant=plant_path,
            conservative=conservative_path,
            rejected=rejected_path,
            uncertain=semantic_dir / "uncertain-semantic.ply",
            output=output_dir / "review",
        )
    else:
        progress("proof-renders-skipped")
        progress("web-viewer-skipped")
        viewer_manifest = {"layers": {}}

    report = {
        "schema_version": 1,
        "source": str(source_path),
        "source_sha256": source_hash,
        "source_point_count": int(len(cloud)),
        "source_opened_read_only": True,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "railing_plan": (
            str(railing_plan_path)
            if railing_plan_path is not None
            else None
        ),
        "railing_plan_sha256": (
            _sha256(railing_plan_path)
            if railing_plan_path is not None
            else None
        ),
        "models": {
            "clipseg": vision["clipseg_model"],
            "sam2": vision["sam2_model"],
            "oneformer": (
                dense_values["model"] if dense_values else None
            ),
        },
        "support_height": support_height,
        "support_cutoff": support_cutoff,
        "counts": {
            "floor_candidate": int(floor_keep.sum()),
            "railing_removed": int((floor_keep & railing_reject).sum()),
            "plant_cleaned": int(final_keep.sum()),
            "plant_conservative": int(final_conservative_keep.sum()),
            "rejected_cleaned": int((~final_keep).sum()),
        },
        "semantic": semantic_report,
        "floor": floor_report,
        "dense_semantic": dense_report,
        "dense_propagation": propagation_report,
        "scene_evidence": scene_report,
        "railing_completion": completion_report,
        "color_correction": color_report,
        "viewer_layers": {
            name: layer["preview_point_count"]
            for name, layer in viewer_manifest["layers"].items()
        },
        "artifacts": {
            "plant": str(plant_path),
            "plant_conservative": str(conservative_path),
            "plant_color_corrected": str(color_path),
            "rejected": str(rejected_path),
            "viewer": (
                str(output_dir / "review" / "viewer.html")
                if build_review_artifacts
                else None
            ),
        },
    }
    _write_json(output_dir / "run-report.json", report)
    progress("complete")
    return report
