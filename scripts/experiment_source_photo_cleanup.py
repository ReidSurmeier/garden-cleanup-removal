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
from plant_cleanup.source_photo_semantic import fuse_source_photo_votes
from railing_removal.floor import remove_uncertain_floor


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", required=True, type=Path)
    parser.add_argument("--source-photo-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    scan_dir = args.scan_dir.resolve()
    source_photo_dir = args.source_photo_dir.resolve()
    output_dir = args.output.resolve()
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
    floor_keep, floor_report = remove_uncertain_floor(
        coordinates,
        normals=normals,
        rgb=rgb,
        decisions=decisions,
        plant_votes=generic_plant,
        background_votes=generic_background,
    )
    plant_votes, background_votes, source_seen = fuse_source_photo_votes(
        source_plant=np.load(source_photo_dir / "plant-votes.npy"),
        source_background=np.load(
            source_photo_dir / "background-votes.npy"
        ),
        fallback_plant=np.load(
            scan_dir / "dense-semantic" / "plant-votes.npy"
        ),
        fallback_background=np.load(
            scan_dir / "dense-semantic" / "background-votes.npy"
        ),
    )
    dense_values = config["dense_semantic"]
    strict, conservative, propagation = propagate_dense_semantic_evidence(
        coordinates,
        normals=normals,
        candidate_mask=floor_keep,
        plant_votes=plant_votes,
        background_votes=background_votes,
        support_plane_coefficients=tuple(
            run_report["semantic"]["support_plane"]["coefficients"]
        ),
        vertical_span=float(config["profile"]["vertical_span"]),
        parameters=DensePropagationParameters(
            **dense_values["propagation"]
        ),
    )
    strict_path = output_dir / "plant-cleaned.ply"
    conservative_path = output_dir / "plant-cleaned-conservative.ply"
    rejected_path = output_dir / "rejected-cleaned.ply"
    write_decision_cloud(cloud, strict_path, strict, decisions)
    write_decision_cloud(
        cloud,
        conservative_path,
        conservative,
        decisions,
    )
    write_decision_cloud(cloud, rejected_path, ~strict, decisions)
    for layer, path in (
        ("source", source_path),
        ("plant", strict_path),
        ("conservative", conservative_path),
        ("rejected", rejected_path),
    ):
        render_cloud_views(
            path,
            output_dir / f"render-{layer}",
            size=int(config["proof_render_size"]),
            point_radius=1,
        )
    report = {
        "schema_version": 1,
        "method": "source-photo-first-with-synthetic-fallback-v1",
        "source": str(source_path),
        "source_opened_read_only": True,
        "baseline_scan_dir": str(scan_dir),
        "source_photo_dir": str(source_photo_dir),
        "source_point_count": int(len(cloud)),
        "source_photo_seen_point_count": int(source_seen.sum()),
        "floor": floor_report,
        "propagation": propagation,
        "counts": {
            "baseline_strict": int(
                run_report["counts"]["plant_cleaned"]
            ),
            "strict": int(strict.sum()),
            "conservative": int(conservative.sum()),
            "rejected": int((~strict).sum()),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
