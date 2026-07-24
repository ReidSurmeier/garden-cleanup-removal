from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

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


@dataclass(frozen=True)
class GroundSurfaceCompletionParameters:
    minimum_object_votes: int = 1
    minimum_surface_seed_points: int = 200
    minimum_surface_span: float = 1.0
    fit_trim_quantile: float = 0.90
    fit_iterations: int = 4
    seed_inlier_distance: float = 0.30
    max_surfaces: int = 6
    ransac_iterations: int = 500
    random_seed: int = 172629
    plane_distance: float = 0.50
    bounds_margin: float = 0.50
    normal_alignment_min: float = 0.55
    strong_plant_margin: int = 2


def _best_seed_plane(
    points: np.ndarray,
    *,
    distance_threshold: float,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(points) < 3:
        return None
    scoring_ids = rng.choice(
        len(points),
        size=min(8_000, len(points)),
        replace=False,
    )
    scoring = points[scoring_ids]
    best: tuple[int, np.ndarray, np.ndarray] | None = None
    for _ in range(iterations):
        first, second, third = points[
            rng.choice(len(points), 3, replace=False)
        ]
        normal = np.cross(second - first, third - first)
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            continue
        normal /= length
        score = int(
            np.count_nonzero(
                np.abs((scoring - first) @ normal)
                <= distance_threshold
            )
        )
        if best is None or score > best[0]:
            best = score, first, normal
    if best is None:
        return None
    _, origin, normal = best
    inliers = (
        np.abs((points - origin) @ normal) <= distance_threshold
    )
    return origin, normal, inliers


def complete_ground_surface_classes(
    coordinates: np.ndarray,
    *,
    normals: np.ndarray,
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray | None,
    class_votes: Mapping[str, np.ndarray],
    class_plant_votes: Mapping[str, np.ndarray],
    parameters: GroundSurfaceCompletionParameters | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Complete raised or sloped ground planes from semantic class evidence."""
    parameters = parameters or GroundSurfaceCompletionParameters()
    coordinates = np.asarray(coordinates, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    point_count = len(coordinates)
    if coordinates.shape != (point_count, 3):
        raise ValueError("coordinates must have shape (point_count, 3)")
    if normals.shape != (point_count, 3):
        raise ValueError("normals must have shape (point_count, 3)")
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.shape != (point_count,):
        raise ValueError(f"candidate_mask must have shape ({point_count},)")
    seed_mask = (
        candidate_mask
        if seed_mask is None
        else np.asarray(seed_mask, dtype=bool)
    )
    if seed_mask.shape != (point_count,):
        raise ValueError(f"seed_mask must have shape ({point_count},)")
    if not class_votes:
        raise ValueError("at least one ground-surface class is required")
    if set(class_votes) != set(class_plant_votes):
        raise ValueError("ground object and plant vote classes must match")
    if parameters.minimum_surface_seed_points < 3:
        raise ValueError("minimum_surface_seed_points must be at least three")
    if not 0.0 < parameters.fit_trim_quantile <= 1.0:
        raise ValueError("fit_trim_quantile must be in (0, 1]")
    if parameters.max_surfaces < 1:
        raise ValueError("max_surfaces must be positive")
    if parameters.ransac_iterations < 1:
        raise ValueError("ransac_iterations must be positive")

    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_normals = normal_lengths > 1e-8
    normalized_normals = normals.copy()
    normalized_normals[valid_normals] /= normal_lengths[valid_normals, None]
    rejected = np.zeros(point_count, dtype=bool)
    class_reports: dict[str, dict[str, Any]] = {}

    for class_id, class_vote_values in class_votes.items():
        object_votes = np.asarray(class_vote_values)
        plant_votes = np.asarray(class_plant_votes[class_id])
        if object_votes.shape != (point_count,):
            raise ValueError(
                f"{class_id} object votes must have shape ({point_count},)"
            )
        if plant_votes.shape != (point_count,):
            raise ValueError(
                f"{class_id} plant votes must have shape ({point_count},)"
            )
        confirmed = (
            seed_mask
            & (object_votes >= parameters.minimum_object_votes)
            & (object_votes >= plant_votes)
        )
        seed_points = coordinates[confirmed]
        class_rejected = np.zeros(point_count, dtype=bool)
        surface_reports: list[dict[str, Any]] = []
        remaining_seed_points = seed_points
        rng = np.random.default_rng(parameters.random_seed)
        strong_plant = plant_votes.astype(np.int16) >= (
            object_votes.astype(np.int16)
            + parameters.strong_plant_margin
        )

        while (
            len(remaining_seed_points)
            >= parameters.minimum_surface_seed_points
            and len(surface_reports) < parameters.max_surfaces
        ):
            fitted = _best_seed_plane(
                remaining_seed_points,
                distance_threshold=parameters.seed_inlier_distance,
                iterations=parameters.ransac_iterations,
                rng=rng,
            )
            if fitted is None:
                break
            _, _, initial_inliers = fitted
            if (
                int(initial_inliers.sum())
                < parameters.minimum_surface_seed_points
            ):
                break
            fit_points = remaining_seed_points[initial_inliers]
            center = np.median(fit_points, axis=0)
            for _ in range(parameters.fit_iterations):
                centered = fit_points - center
                covariance = centered.T @ centered / len(fit_points)
                values, vectors = np.linalg.eigh(covariance)
                normal = vectors[:, np.argmin(values)]
                distances = np.abs((fit_points - center) @ normal)
                cutoff = np.quantile(
                    distances,
                    parameters.fit_trim_quantile,
                )
                fit_points = fit_points[distances <= cutoff]
                center = np.median(fit_points, axis=0)

            if len(fit_points) < 3:
                remaining_seed_points = remaining_seed_points[
                    ~initial_inliers
                ]
                continue
            centered = fit_points - center
            covariance = centered.T @ centered / len(fit_points)
            values, vectors = np.linalg.eigh(covariance)
            order = np.argsort(values)
            normal = vectors[:, order[0]]
            axes = vectors[:, order[1:]]
            final_inliers = (
                np.abs((remaining_seed_points - center) @ normal)
                <= parameters.seed_inlier_distance
            )
            surface_seed_points = remaining_seed_points[final_inliers]
            remaining_seed_points = remaining_seed_points[~final_inliers]
            if (
                len(surface_seed_points)
                < parameters.minimum_surface_seed_points
            ):
                continue
            seed_projection = (surface_seed_points - center) @ axes
            bounds = np.quantile(
                seed_projection,
                (0.005, 0.995),
                axis=0,
            )
            spans = bounds[1] - bounds[0]
            fit_distances = np.abs((fit_points - center) @ normal)
            planar = (
                float(np.quantile(fit_distances, 0.95))
                <= parameters.plane_distance
            )
            if (
                planar
                and float(spans.max()) >= parameters.minimum_surface_span
            ):
                projected = (coordinates - center) @ axes
                inside = np.all(
                    (
                        projected
                        >= bounds[0] - parameters.bounds_margin
                    )
                    & (
                        projected
                        <= bounds[1] + parameters.bounds_margin
                    ),
                    axis=1,
                )
                distance = np.abs((coordinates - center) @ normal)
                alignment = np.abs(normalized_normals @ normal)
                class_exclusive_ground = (
                    object_votes >= parameters.minimum_object_votes
                ) & (object_votes > plant_votes)
                surface_rejected = (
                    candidate_mask
                    & inside
                    & (distance <= parameters.plane_distance)
                    & (
                        (
                            valid_normals
                            & (
                                alignment
                                >= parameters.normal_alignment_min
                            )
                        )
                        | class_exclusive_ground
                    )
                    & ~strong_plant
                )
                newly_completed = surface_rejected & ~class_rejected
                class_rejected |= surface_rejected
                surface_reports.append(
                    {
                        "center": center.tolist(),
                        "normal": normal.tolist(),
                        "spans": spans.tolist(),
                        "seed_point_count": int(
                            len(surface_seed_points)
                        ),
                        "fit_point_count": int(len(fit_points)),
                        "seed_distance_95": float(
                            np.quantile(fit_distances, 0.95)
                        ),
                        "completed_point_count": int(
                            surface_rejected.sum()
                        ),
                        "newly_completed_point_count": int(
                            newly_completed.sum()
                        ),
                    }
                )

        rejected |= class_rejected
        class_reports[class_id] = {
            "schema_version": 1,
            "parameters": asdict(parameters),
            "candidate_point_count": int(candidate_mask.sum()),
            "seed_candidate_point_count": int(seed_mask.sum()),
            "confirmed_seed_count": int(confirmed.sum()),
            "accepted_surface_count": len(surface_reports),
            "completed_point_count": int(class_rejected.sum()),
            "surfaces": surface_reports,
        }

    return rejected, {
        "schema_version": 1,
        "strategy": "semantic-seeded-ground-surfaces-v2",
        "class_count": len(class_reports),
        "completed_point_count": int(rejected.sum()),
        "classes": class_reports,
    }


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
