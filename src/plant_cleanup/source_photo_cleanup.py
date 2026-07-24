from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from plant_cleanup.cloud_render import render_cloud_views
from plant_cleanup.color_correct import (
    ColorParameters,
    correct_cloud_colors,
)
from plant_cleanup.dense_semantic import (
    DensePropagationParameters,
    propagate_dense_semantic_evidence,
)
from plant_cleanup.geometry_cleanup import write_decision_cloud
from plant_cleanup.plyio import read_cloud
from plant_cleanup.source_photo_semantic import fuse_source_photo_votes
from railing_removal.floor import remove_uncertain_floor
from railing_removal.full_pipeline import _build_viewer


Progress = Callable[[str], None]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def run_source_photo_cleanup(
    baseline_scan_dir: Path,
    source_photo_dir: Path,
    output_dir: Path,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build a publishable cleanup using calibrated real-photo evidence."""

    baseline_scan_dir = baseline_scan_dir.resolve()
    source_photo_dir = source_photo_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    progress = progress or (lambda _: None)

    baseline_report = _json(baseline_scan_dir / "run-report.json")
    if not baseline_report.get("source_opened_read_only"):
        raise ValueError("baseline lacks read-only source provenance")
    source_photo_report = _json(source_photo_dir / "report.json")
    if not source_photo_report.get("project_opened_read_only"):
        raise ValueError("source-photo evidence lacks read-only provenance")
    config_path = Path(baseline_report["config"])
    config = _json(config_path)
    source_path = Path(baseline_report["source"])

    progress("load-evidence")
    cloud = read_cloud(source_path)
    coordinates = np.column_stack(
        (cloud["x"], cloud["y"], cloud["z"])
    )
    normals = np.column_stack(
        (cloud["nx"], cloud["ny"], cloud["nz"])
    )
    rgb = np.column_stack(
        (cloud["red"], cloud["green"], cloud["blue"])
    )
    final_decisions = baseline_scan_dir / "final" / "decision-codes.npy"
    decisions = np.load(
        final_decisions
        if final_decisions.is_file()
        else baseline_scan_dir / "semantic" / "decision-codes.npy"
    )
    generic_plant = np.load(
        baseline_scan_dir / "vision-sam2" / "plant-votes.npy"
    )
    generic_background = np.load(
        baseline_scan_dir / "vision-sam2" / "planter-votes.npy"
    )

    progress("uncertain-floor-removal")
    floor_keep, floor_report = remove_uncertain_floor(
        coordinates,
        normals=normals,
        rgb=rgb,
        decisions=decisions,
        plant_votes=generic_plant,
        background_votes=generic_background,
    )

    progress("source-photo-fusion")
    source_plant = np.load(source_photo_dir / "plant-votes.npy")
    source_background = np.load(
        source_photo_dir / "background-votes.npy"
    )
    plant_votes, background_votes, source_seen = fuse_source_photo_votes(
        source_plant=source_plant,
        source_background=source_background,
        fallback_plant=np.load(
            baseline_scan_dir
            / "dense-semantic"
            / "plant-votes.npy"
        ),
        fallback_background=np.load(
            baseline_scan_dir
            / "dense-semantic"
            / "background-votes.npy"
        ),
    )

    progress("dense-structural-propagation")
    dense_values = config["dense_semantic"]
    strict, conservative, propagation = propagate_dense_semantic_evidence(
        coordinates,
        normals=normals,
        candidate_mask=floor_keep,
        plant_votes=plant_votes,
        background_votes=background_votes,
        support_plane_coefficients=tuple(
            baseline_report["semantic"]["support_plane"]["coefficients"]
        ),
        vertical_span=float(config["profile"]["vertical_span"]),
        parameters=DensePropagationParameters(
            **dense_values["propagation"]
        ),
    )

    progress("write-versioned-clouds")
    final_dir = output_dir / "final"
    final_dir.mkdir()
    plant_path = final_dir / "plant-cleaned.ply"
    conservative_path = final_dir / "plant-cleaned-conservative.ply"
    rejected_path = final_dir / "rejected-cleaned.ply"
    color_path = final_dir / "plant-cleaned-color-corrected.ply"
    write_decision_cloud(cloud, plant_path, strict, decisions)
    write_decision_cloud(
        cloud,
        conservative_path,
        conservative,
        decisions,
    )
    write_decision_cloud(cloud, rejected_path, ~strict, decisions)

    progress("color-correction")
    color_report = correct_cloud_colors(
        plant_path,
        color_path,
        ColorParameters(**config["color_correction"]),
    )
    _write_json(final_dir / "color-report.json", color_report)

    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir()
    unseen_path = evidence_dir / "source-photo-unseen.ply"
    write_decision_cloud(cloud, unseen_path, ~source_seen, decisions)

    progress("proof-renders")
    proof_paths = (
        ("source", source_path),
        ("plant", color_path),
        ("conservative", conservative_path),
        ("rejected", rejected_path),
    )
    for name, path in proof_paths:
        render_dir = final_dir / f"render-{name}"
        proof = render_cloud_views(
            path,
            render_dir,
            size=int(config["proof_render_size"]),
            point_radius=1,
        )
        _write_json(render_dir / "render-report.json", proof)

    progress("web-viewer")
    viewer_manifest = _build_viewer(
        source=source_path,
        previous=baseline_scan_dir / "final" / "plant-cleaned.ply",
        plant=color_path,
        conservative=conservative_path,
        rejected=rejected_path,
        uncertain=unseen_path,
        output=output_dir / "review",
    )

    observed_plant = source_plant > source_background
    observed_background = source_background > source_plant
    report = {
        "schema_version": 1,
        "method": "calibrated-source-photo-cleanup-v2",
        "source": str(source_path),
        "source_point_count": int(len(cloud)),
        "source_opened_read_only": True,
        "baseline_scan_dir": str(baseline_scan_dir),
        "config": str(config_path),
        "source_photo_dir": str(source_photo_dir),
        "models": {
            "source_photo_semantic": source_photo_report.get("model"),
            "baseline": baseline_report.get("models"),
        },
        "source_photo": {
            "camera_count": source_photo_report.get("camera_count"),
            "seen_point_count": int(source_seen.sum()),
            "unseen_point_count": int((~source_seen).sum()),
            "plant_point_count": int(observed_plant.sum()),
            "background_point_count": int(observed_background.sum()),
        },
        "floor": floor_report,
        "dense_propagation": propagation,
        "color_correction": color_report,
        "counts": {
            "floor_candidate": int(floor_keep.sum()),
            "plant_cleaned": int(strict.sum()),
            "plant_conservative": int(conservative.sum()),
            "rejected_cleaned": int((~strict).sum()),
        },
        "viewer_layers": {
            name: layer["preview_point_count"]
            for name, layer in viewer_manifest["layers"].items()
        },
        "artifacts": {
            "plant": str(plant_path),
            "plant_conservative": str(conservative_path),
            "plant_color_corrected": str(color_path),
            "rejected": str(rejected_path),
            "viewer": str(output_dir / "review" / "viewer.html"),
        },
    }
    _write_json(output_dir / "run-report.json", report)
    progress("complete")
    return report
