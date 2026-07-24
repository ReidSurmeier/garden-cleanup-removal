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
    grow_floor_components: bool = False
    strong_plant_margin: int = 2


def remove_uncertain_floor(
    coordinates: np.ndarray,
    *,
    normals: np.ndarray,
    rgb: np.ndarray,
    decisions: np.ndarray,
    plant_votes: np.ndarray,
    background_votes: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    parameters: FloorRemovalParameters | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove coplanar floor geometry seeded by the uncertain decision layer.

    ``candidate_mask`` supports a second, post-semantic pass without restoring
    anything rejected by the first pass. The uncertain decision layer remains
    the only source of floor seeds.
    """

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
    if candidate_mask is not None:
        arrays["candidate_mask"] = np.asarray(candidate_mask, dtype=bool)
    for name, values in arrays.items():
        if values.shape != (point_count,):
            raise ValueError(f"{name} must have shape ({point_count},)")
    decisions = arrays["decisions"]
    plant_votes = arrays["plant_votes"]
    background_votes = arrays["background_votes"]
    candidate_mask = arrays.get("candidate_mask")

    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_normals = normal_lengths > 1e-8
    normalized_normals = normals.copy()
    normalized_normals[valid_normals] /= normal_lengths[valid_normals, None]
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    keep_before_floor = (
        candidate_mask.copy()
        if candidate_mask is not None
        else (
            (decisions == 1)
            | ((decisions == 6) & (plant_votes > background_votes))
        )
    )
    keep = keep_before_floor.copy()
    uncertain_rows = np.flatnonzero(decisions == 5)
    uncertain = coordinates[uncertain_rows]
    floor = np.zeros(point_count, dtype=bool)
    grown_floor = np.zeros(point_count, dtype=bool)
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
            component_uncertain = point_labels == label
            points = uncertain[component_uncertain]
            component_seed_rows = uncertain_rows[component_uncertain]
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
            component_grown = np.zeros(point_count, dtype=bool)
            if parameters.grow_floor_components:
                strong_plant = plant_votes.astype(np.int16) >= (
                    background_votes.astype(np.int16)
                    + parameters.strong_plant_margin
                )
                # A dense vision model can call flat brown mulch "grass".
                # Protect its plant vote only when color or surface orientation
                # independently supports plant structure. Brown geometry that
                # is aligned to the seeded floor plane remains eligible.
                independent_plant_structure = (
                    (excess_green >= parameters.excess_green_max)
                    | (alignment < parameters.normal_alignment_min)
                )
                protected_plant = strong_plant & independent_plant_structure
                growth_candidates = (
                    keep_before_floor | (decisions == 5)
                    if candidate_mask is not None
                    else np.isin(decisions, [1, 5, 6])
                )
                eligible = (
                    growth_candidates
                    & (distance <= parameters.plane_distance)
                    & valid_normals
                    & (alignment >= parameters.normal_alignment_min)
                    & ~protected_plant
                )
                eligible[component_seed_rows] = True
                rows = np.flatnonzero(eligible)
                if len(rows):
                    projected = (coordinates[rows] - center) @ axes
                    lower_2d = projected.min(axis=0)
                    projected_voxels = np.floor(
                        (projected - lower_2d) / parameters.voxel_size
                    ).astype(np.int32)
                    shape_2d = tuple(
                        (projected_voxels.max(axis=0) + 1).tolist()
                    )
                    grid_cells = int(np.prod(shape_2d, dtype=np.int64))
                    if grid_cells > 100_000_000:
                        raise ValueError(
                            "adaptive floor grid would contain "
                            f"{grid_cells:,} cells; increase voxel_size"
                        )
                    occupancy_2d = np.zeros(shape_2d, dtype=bool)
                    occupancy_2d[tuple(projected_voxels.T)] = True
                    labels_2d, _ = ndimage.label(
                        occupancy_2d,
                        structure=np.ones((3, 3), dtype=bool),
                    )
                    row_labels = labels_2d[tuple(projected_voxels.T)]
                    seed_source = np.zeros(point_count, dtype=bool)
                    seed_source[component_seed_rows] = True
                    seed_labels = np.unique(
                        row_labels[seed_source[rows]]
                    )
                    seed_labels = seed_labels[seed_labels != 0]
                    component_grown[rows] = np.isin(
                        row_labels,
                        seed_labels,
                    )
                    grown_floor |= component_grown
            component_reports.append(
                {
                    "label": int(label),
                    "seed_point_count": int(len(points)),
                    "matched_source_point_count": int(component_floor.sum()),
                    "grown_source_point_count": int(component_grown.sum()),
                    "normal": [float(value) for value in normal],
                }
            )

    floor |= grown_floor
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
        "grown_floor_point_count": int(grown_floor.sum()),
        "floor_components": component_reports,
    }
    return keep, report
