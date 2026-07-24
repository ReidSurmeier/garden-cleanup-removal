from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class FloorRemovalParameters:
    voxel_size: float = 0.30
    minimum_component_points: int = 500
    component_fraction: float = 0.02
    plane_distance: float = 0.50
    bounds_margin: float = 0.50
    normal_alignment_min: float = 0.55
    excess_green_max: float = 10.0


def remove_uncertain_floor(
    coordinates: np.ndarray,
    *,
    normals: np.ndarray,
    rgb: np.ndarray,
    decisions: np.ndarray,
    plant_votes: np.ndarray,
    background_votes: np.ndarray,
    parameters: FloorRemovalParameters | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove coplanar floor geometry seeded by the uncertain decision layer."""

    parameters = parameters or FloorRemovalParameters()
    coordinates = np.asarray(coordinates, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    point_count = len(coordinates)
    if coordinates.shape != (point_count, 3):
        raise ValueError("coordinates must have shape (point_count, 3)")
    if normals.shape != (point_count, 3):
        raise ValueError("normals must have shape (point_count, 3)")
    if rgb.shape != (point_count, 3):
        raise ValueError("rgb must have shape (point_count, 3)")
    arrays = {
        "decisions": np.asarray(decisions),
        "plant_votes": np.asarray(plant_votes),
        "background_votes": np.asarray(background_votes),
    }
    for name, values in arrays.items():
        if values.shape != (point_count,):
            raise ValueError(f"{name} must have shape ({point_count},)")
    decisions = arrays["decisions"]
    plant_votes = arrays["plant_votes"]
    background_votes = arrays["background_votes"]

    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_normals = normal_lengths > 1e-8
    normalized_normals = normals.copy()
    normalized_normals[valid_normals] /= normal_lengths[valid_normals, None]
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    keep_before_floor = (decisions == 1) | (
        (decisions == 6) & (plant_votes > background_votes)
    )
    keep = keep_before_floor.copy()
    uncertain = coordinates[decisions == 5]
    floor = np.zeros(point_count, dtype=bool)
    component_reports: list[dict[str, Any]] = []

    if len(uncertain):
        lower = uncertain.min(axis=0)
        voxels = np.floor(
            (uncertain - lower) / parameters.voxel_size
        ).astype(np.int32)
        shape = tuple((voxels.max(axis=0) + 1).tolist())
        occupancy = np.zeros(shape, dtype=bool)
        occupancy[tuple(voxels.T)] = True
        labels, _ = ndimage.label(
            occupancy,
            structure=np.ones((3, 3, 3), dtype=bool),
        )
        point_labels = labels[tuple(voxels.T)]
        counts = np.bincount(point_labels)
        counts[0] = 0
        minimum_component = max(
            parameters.minimum_component_points,
            int(len(uncertain) * parameters.component_fraction),
        )

        for label in np.flatnonzero(counts >= minimum_component):
            points = uncertain[point_labels == label]
            center = np.median(points, axis=0)
            fit_points = points
            for _ in range(3):
                centered = fit_points - center
                covariance = centered.T @ centered / len(fit_points)
                values, vectors = np.linalg.eigh(covariance)
                normal = vectors[:, np.argmin(values)]
                distances = np.abs((points - center) @ normal)
                fit_points = points[distances <= np.quantile(distances, 0.9)]
                center = np.median(fit_points, axis=0)

            centered = fit_points - center
            covariance = centered.T @ centered / len(fit_points)
            values, vectors = np.linalg.eigh(covariance)
            order = np.argsort(values)
            normal = vectors[:, order[0]]
            axes = vectors[:, order[1:]]
            projected_seed = (points - center) @ axes
            bounds = np.quantile(projected_seed, [0.005, 0.995], axis=0)
            projected_all = (coordinates - center) @ axes
            inside = np.all(
                (projected_all >= bounds[0] - parameters.bounds_margin)
                & (projected_all <= bounds[1] + parameters.bounds_margin),
                axis=1,
            )
            distance = np.abs((coordinates - center) @ normal)
            alignment = np.abs(normalized_normals @ normal)
            component_floor = (
                inside
                & (distance <= parameters.plane_distance)
                & valid_normals
                & (alignment >= parameters.normal_alignment_min)
                & (excess_green < parameters.excess_green_max)
            )
            floor |= component_floor
            component_reports.append(
                {
                    "label": int(label),
                    "seed_point_count": int(len(points)),
                    "matched_source_point_count": int(component_floor.sum()),
                    "normal": [float(value) for value in normal],
                }
            )

    keep &= ~floor
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "uncertain-seeded-coplanar-floor-rejection-v1",
        "parameters": asdict(parameters),
        "source_point_count": point_count,
        "plant_point_count": int(keep.sum()),
        "rejected_point_count": int((~keep).sum()),
        "uncertain_rejected_point_count": int(np.count_nonzero(decisions == 5)),
        "demoted_geometry_recovery_point_count": int(
            np.count_nonzero(
                (decisions == 6) & ~(plant_votes > background_votes)
            )
        ),
        "coplanar_points_removed_from_candidate": int(
            np.count_nonzero(keep_before_floor & floor)
        ),
        "floor_components": component_reports,
    }
    return keep, report
