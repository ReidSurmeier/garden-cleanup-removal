from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from plant_cleanup.plyio import read_cloud  # noqa: E402


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(quantile): float(value)
        for quantile, value in zip(
            (0.0, 0.5, 0.9, 0.95, 0.99, 1.0),
            np.quantile(values, (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)),
        )
    }


def _coordinate_quantiles(points: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        axis: _quantiles(points[:, index])
        for index, axis in enumerate(("x", "y", "z"))
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only geometry diagnostics for a planned scene class."
    )
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("class_id")
    args = parser.parse_args()

    scan_dir = args.scan_dir.resolve()
    run_report = json.loads(
        (scan_dir / "run-report.json").read_text(encoding="utf-8")
    )
    cloud = read_cloud(Path(run_report["source"]))
    coordinates = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(
        np.float64,
        copy=False,
    )
    rgb = np.column_stack((cloud["red"], cloud["green"], cloud["blue"])).astype(
        np.float64,
        copy=False,
    )
    normals = np.column_stack((cloud["nx"], cloud["ny"], cloud["nz"])).astype(
        np.float64,
        copy=False,
    )
    fused = scan_dir / "scene-evidence" / "fused"
    object_votes = np.load(fused / f"{args.class_id}-votes.npy")
    plant_votes = np.load(fused / f"{args.class_id}-plant-votes.npy")
    seed_mask = (object_votes >= 1) & (object_votes > plant_votes)
    manual_path = fused / f"{args.class_id}-manual-seed-mask.npy"
    manual_mask = (
        np.asarray(np.load(manual_path), dtype=bool)
        if manual_path.is_file()
        else np.zeros(len(cloud), dtype=bool)
    )
    seed_mask |= manual_mask
    seed_points = coordinates[seed_mask]
    if len(seed_points) < 3:
        raise ValueError("fewer than three class-exclusive evidence points")

    fit_points = seed_points
    center = np.median(fit_points, axis=0)
    for _ in range(4):
        centered = fit_points - center
        covariance = centered.T @ centered / len(fit_points)
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, np.argmin(values)]
        distances = np.abs((fit_points - center) @ normal)
        fit_points = fit_points[
            distances <= np.quantile(distances, 0.9)
        ]
        center = np.median(fit_points, axis=0)

    centered = fit_points - center
    covariance = centered.T @ centered / len(fit_points)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    normal = vectors[:, order[0]]
    axes = vectors[:, order[1:]]
    seed_distance = np.abs((seed_points - center) @ normal)
    seed_projection = (seed_points - center) @ axes
    bounds = np.quantile(seed_projection, (0.005, 0.995), axis=0)
    projected = (coordinates - center) @ axes
    inside = np.all(
        (projected >= bounds[0] - 0.5)
        & (projected <= bounds[1] + 0.5),
        axis=1,
    )
    distance = np.abs((coordinates - center) @ normal)
    candidate = inside & (distance <= 0.4)
    lengths = np.linalg.norm(normals, axis=1)
    valid_normals = lengths > 1e-8
    normalized = normals.copy()
    normalized[valid_normals] /= lengths[valid_normals, None]
    alignment = np.abs(normalized @ normal)
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    saturation = (rgb.max(axis=1) - rgb.min(axis=1)) / np.maximum(
        rgb.max(axis=1),
        1.0,
    )
    structural_seed_mask = (
        seed_mask
        & (excess_green <= 5.0)
        & (saturation <= 0.55)
    )
    structural_seed_points = coordinates[structural_seed_mask]

    report = {
        "schema_version": 1,
        "source_opened_read_only": True,
        "class_id": args.class_id,
        "source_point_count": len(cloud),
        "seed_point_count": int(seed_mask.sum()),
        "manual_seed_point_count": int(manual_mask.sum()),
        "seed_coordinate_quantiles": _coordinate_quantiles(seed_points),
        "structural_seed_point_count": int(structural_seed_mask.sum()),
        "structural_seed_coordinate_quantiles": _coordinate_quantiles(
            structural_seed_points
        ),
        "fit_point_count": len(fit_points),
        "plane_center": center.tolist(),
        "plane_normal": normal.tolist(),
        "plane_eigenvalues": values[order].tolist(),
        "seed_distance_quantiles": _quantiles(seed_distance),
        "candidate_within_0.4_count": int(candidate.sum()),
        "candidate_alignment_quantiles": _quantiles(alignment[candidate]),
        "candidate_excess_green_quantiles": _quantiles(
            excess_green[candidate]
        ),
        "candidate_strong_plant_count": int(
            np.count_nonzero(
                candidate & (plant_votes.astype(np.int16) >= (
                    object_votes.astype(np.int16) + 3
                ))
            )
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
