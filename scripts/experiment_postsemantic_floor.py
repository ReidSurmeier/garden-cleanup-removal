from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plant_cleanup.cloud_render import render_cloud_views
from plant_cleanup.dense_semantic import (
    DensePropagationParameters,
    propagate_dense_semantic_evidence,
)
from plant_cleanup.geometry_cleanup import write_decision_cloud
from plant_cleanup.plyio import read_cloud
from railing_removal.floor import FloorRemovalParameters, remove_uncertain_floor


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _variants(vertical_span: float) -> dict[str, FloorRemovalParameters]:
    return {
        "default-growth": FloorRemovalParameters(
            grow_floor_components=True,
        ),
        "scale-safe": FloorRemovalParameters(
            voxel_size=max(vertical_span * 0.015, 0.03),
            plane_distance=max(vertical_span * 0.020, 0.05),
            bounds_margin=max(vertical_span * 0.030, 0.05),
            normal_alignment_min=0.65,
            grow_floor_components=True,
        ),
        "scale-precise": FloorRemovalParameters(
            voxel_size=max(vertical_span * 0.010, 0.02),
            plane_distance=max(vertical_span * 0.0125, 0.03),
            bounds_margin=max(vertical_span * 0.020, 0.04),
            normal_alignment_min=0.72,
            grow_floor_components=True,
        ),
    }


def run_experiment(scan_dir: Path, output_dir: Path) -> dict:
    scan_dir = scan_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    run_report = _json(scan_dir / "run-report.json")
    config = _json(Path(run_report["config"]))
    source_path = Path(run_report["source"])
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
    decisions = np.load(scan_dir / "semantic" / "decision-codes.npy")
    generic_plant = np.load(
        scan_dir / "vision-sam2" / "plant-votes.npy"
    )
    generic_background = np.load(
        scan_dir / "vision-sam2" / "planter-votes.npy"
    )
    floor_keep, _ = remove_uncertain_floor(
        coordinates,
        normals=normals,
        rgb=rgb,
        decisions=decisions,
        plant_votes=generic_plant,
        background_votes=generic_background,
    )
    dense_values = config["dense_semantic"]
    dense_plant = np.load(
        scan_dir / "dense-semantic" / "plant-votes.npy"
    )
    dense_background = np.load(
        scan_dir / "dense-semantic" / "background-votes.npy"
    )
    strict, conservative, propagation = propagate_dense_semantic_evidence(
        coordinates,
        normals=normals,
        candidate_mask=floor_keep,
        plant_votes=dense_plant,
        background_votes=dense_background,
        support_plane_coefficients=tuple(
            run_report["semantic"]["support_plane"]["coefficients"]
        ),
        vertical_span=float(config["profile"]["vertical_span"]),
        parameters=DensePropagationParameters(
            **dense_values["propagation"]
        ),
    )
    experiment = {
        "schema_version": 1,
        "source": str(source_path),
        "source_opened_read_only": True,
        "baseline": {
            "strict_point_count": int(strict.sum()),
            "conservative_point_count": int(conservative.sum()),
            "propagation": propagation,
        },
        "variants": {},
    }
    for name, parameters in _variants(
        float(config["profile"]["vertical_span"])
    ).items():
        variant_dir = output_dir / name
        variant_dir.mkdir()
        post_keep, post_report = remove_uncertain_floor(
            coordinates,
            normals=normals,
            rgb=rgb,
            decisions=decisions,
            plant_votes=dense_plant,
            background_votes=dense_background,
            candidate_mask=conservative,
            parameters=parameters,
        )
        removed = conservative & ~post_keep
        strict_keep = strict & ~removed
        strict_path = variant_dir / "plant-cleaned.ply"
        conservative_path = variant_dir / "plant-cleaned-conservative.ply"
        rejected_path = variant_dir / "rejected-cleaned.ply"
        write_decision_cloud(cloud, strict_path, strict_keep, decisions)
        write_decision_cloud(
            cloud,
            conservative_path,
            post_keep,
            decisions,
        )
        write_decision_cloud(
            cloud,
            rejected_path,
            ~strict_keep,
            decisions,
        )
        for layer, path in (
            ("plant", strict_path),
            ("conservative", conservative_path),
            ("rejected", rejected_path),
        ):
            render_cloud_views(
                path,
                variant_dir / f"render-{layer}",
                size=int(config["proof_render_size"]),
                point_radius=1,
            )
        variant_report = {
            "parameters": post_report["parameters"],
            "strict_point_count": int(strict_keep.sum()),
            "conservative_point_count": int(post_keep.sum()),
            "removed_from_strict": int(
                np.count_nonzero(strict & ~strict_keep)
            ),
            "removed_from_conservative": int(removed.sum()),
            "floor": post_report,
        }
        _write_json(variant_dir / "report.json", variant_report)
        experiment["variants"][name] = variant_report
    _write_json(output_dir / "experiment-report.json", experiment)
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bounded post-semantic floor-growth variants."
    )
    parser.add_argument("--scan-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_experiment(args.scan_dir, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
