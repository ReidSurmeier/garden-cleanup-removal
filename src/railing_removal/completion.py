from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class LineCompletionParameters:
    minimum_railing_votes: int = 1
    seed_excess_green_max: float = 5.0
    seed_saturation_max: float = 0.55
    seed_distance: float = 0.09
    completion_radius: float = 0.35
    completion_margin: float = 0.25
    completion_excess_green_max: float = 35.0
    completion_saturation_max: float = 0.45
    strong_plant_margin: int = 3
    minimum_seed_height_above_support: float | None = None
    minimum_line_seed_points: int = 15
    minimum_line_length: float = 0.70
    max_lines: int = 40
    ransac_iterations: int = 3_000
    random_seed: int = 172629


@dataclass(frozen=True)
class RigidSurfaceCompletionParameters:
    minimum_object_votes: int = 1
    seed_excess_green_max: float = 5.0
    seed_saturation_max: float = 0.55
    minimum_surface_seed_points: int = 200
    minimum_surface_span: float = 1.0
    fit_trim_quantile: float = 0.90
    fit_iterations: int = 4
    plane_distance: float = 0.40
    bounds_margin: float = 0.50
    completion_excess_green_max: float = 35.0
    completion_saturation_max: float = 0.45
    strong_plant_margin: int = 3


@dataclass(frozen=True)
class RigidComponentCompletionParameters:
    minimum_object_votes: int = 1
    seed_excess_green_max: float = 5.0
    seed_saturation_max: float = 0.55
    minimum_seed_height_above_support: float = 0.80
    minimum_completion_height_above_support: float = 0.20
    voxel_size: float = 0.30
    bounds_margin: float = 0.50
    completion_excess_green_max: float = 10.0
    completion_saturation_max: float = 0.45
    strong_plant_margin: int = 3
    maximum_grid_voxels: int = 100_000_000


def _validate_vector(name: str, values: np.ndarray, point_count: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (point_count,):
        raise ValueError(f"{name} must have shape ({point_count},)")
    return values


def _best_line(
    points: np.ndarray,
    *,
    distance_threshold: float,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None
    scoring_ids = rng.choice(
        len(points),
        size=min(8_000, len(points)),
        replace=False,
    )
    scoring = points[scoring_ids]
    best: tuple[int, np.ndarray, np.ndarray] | None = None
    for _ in range(iterations):
        first, second = points[rng.choice(len(points), 2, replace=False)]
        direction = second - first
        length = float(np.linalg.norm(direction))
        if length < 0.3:
            continue
        direction /= length
        delta = scoring - first
        distances = np.linalg.norm(
            delta - np.outer(delta @ direction, direction),
            axis=1,
        )
        score = int(np.count_nonzero(distances <= distance_threshold))
        if best is None or score > best[0]:
            best = score, first, direction
    if best is None:
        return None
    _, origin, direction = best
    delta = points - origin
    distances = np.linalg.norm(
        delta - np.outer(delta @ direction, direction),
        axis=1,
    )
    return origin, direction, distances <= distance_threshold


def complete_railing_lines(
    coordinates: np.ndarray,
    *,
    rgb: np.ndarray,
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray | None = None,
    railing_votes: np.ndarray,
    plant_votes: np.ndarray,
    support_height: float | None = None,
    parameters: LineCompletionParameters | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Complete rigid rail lines from sparse class-exclusive semantic evidence."""

    parameters = parameters or LineCompletionParameters()
    coordinates = np.asarray(coordinates, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (point_count, 3)")
    point_count = len(coordinates)
    if rgb.shape != (point_count, 3):
        raise ValueError(f"rgb must have shape ({point_count}, 3)")
    candidate_mask = _validate_vector(
        "candidate_mask", candidate_mask, point_count
    ).astype(bool, copy=False)
    seed_mask = (
        candidate_mask
        if seed_mask is None
        else _validate_vector("seed_mask", seed_mask, point_count).astype(
            bool,
            copy=False,
        )
    )
    railing_votes = _validate_vector(
        "railing_votes", railing_votes, point_count
    )
    plant_votes = _validate_vector("plant_votes", plant_votes, point_count)
    if parameters.minimum_line_seed_points < 2:
        raise ValueError("minimum_line_seed_points must be at least two")
    if (
        parameters.minimum_seed_height_above_support is not None
        and parameters.minimum_seed_height_above_support < 0
    ):
        raise ValueError(
            "minimum_seed_height_above_support must be non-negative"
        )
    if (
        parameters.minimum_seed_height_above_support is not None
        and support_height is None
    ):
        raise ValueError(
            "support_height is required for a relative seed height gate"
        )

    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    saturation = (rgb.max(axis=1) - rgb.min(axis=1)) / np.maximum(
        rgb.max(axis=1),
        1.0,
    )
    confirmed = (
        seed_mask
        & (railing_votes >= parameters.minimum_railing_votes)
        & (railing_votes > plant_votes)
    )
    structural_seeds = (
        confirmed
        & (excess_green <= parameters.seed_excess_green_max)
        & (saturation <= parameters.seed_saturation_max)
    )
    height_eligible_seeds = structural_seeds.copy()
    if parameters.minimum_seed_height_above_support is not None:
        assert support_height is not None
        height_eligible_seeds &= coordinates[:, 2] >= (
            support_height
            + parameters.minimum_seed_height_above_support
        )
    remaining_ids = np.flatnonzero(height_eligible_seeds)
    rejected = np.zeros(point_count, dtype=bool)
    rng = np.random.default_rng(parameters.random_seed)
    line_reports: list[dict[str, Any]] = []

    while (
        len(remaining_ids) >= parameters.minimum_line_seed_points
        and len(line_reports) < parameters.max_lines
    ):
        points = coordinates[remaining_ids]
        fitted = _best_line(
            points,
            distance_threshold=parameters.seed_distance,
            iterations=parameters.ransac_iterations,
            rng=rng,
        )
        if fitted is None:
            break
        origin, direction, inliers = fitted
        seed_count = int(inliers.sum())
        if seed_count < parameters.minimum_line_seed_points:
            break
        seed_points = points[inliers]
        seed_projection = (seed_points - origin) @ direction
        bounds = np.quantile(seed_projection, [0.01, 0.99])
        line_length = float(bounds[1] - bounds[0])
        remaining_ids = remaining_ids[~inliers]
        if line_length < parameters.minimum_line_length:
            continue

        delta = coordinates - origin
        projection = delta @ direction
        distance = np.linalg.norm(
            delta - np.outer(projection, direction),
            axis=1,
        )
        color_eligible = (
            excess_green <= parameters.completion_excess_green_max
        ) | (saturation <= parameters.completion_saturation_max)
        strong_plant = plant_votes.astype(np.int16) >= (
            railing_votes.astype(np.int16) + parameters.strong_plant_margin
        )
        completion = (
            candidate_mask
            & (distance <= parameters.completion_radius)
            & (projection >= bounds[0] - parameters.completion_margin)
            & (projection <= bounds[1] + parameters.completion_margin)
            & color_eligible
            & ~strong_plant
        )
        new_points = completion & ~rejected
        rejected |= completion
        line_reports.append(
            {
                "origin": [float(value) for value in origin],
                "direction": [float(value) for value in direction],
                "length": line_length,
                "seed_point_count": seed_count,
                "completed_point_count": int(completion.sum()),
                "newly_completed_point_count": int(new_points.sum()),
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "parameters": asdict(parameters),
        "candidate_point_count": int(candidate_mask.sum()),
        "seed_candidate_point_count": int(seed_mask.sum()),
        "confirmed_seed_count": int(confirmed.sum()),
        "structural_seed_count": int(structural_seeds.sum()),
        "height_eligible_seed_count": int(height_eligible_seeds.sum()),
        "support_height": (
            None if support_height is None else float(support_height)
        ),
        "accepted_line_count": len(line_reports),
        "completed_point_count": int(rejected.sum()),
        "lines": line_reports,
    }
    return rejected, report


def complete_rigid_line_classes(
    coordinates: np.ndarray,
    *,
    rgb: np.ndarray,
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray | None = None,
    class_votes: Mapping[str, np.ndarray],
    class_plant_votes: Mapping[str, np.ndarray],
    support_height: float | None = None,
    parameters: LineCompletionParameters | None = None,
    parameters_by_class: Mapping[
        str, LineCompletionParameters
    ] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Complete every planned rigid-line class and preserve class provenance."""
    if not class_votes:
        raise ValueError("at least one rigid-line class is required")
    if set(class_votes) != set(class_plant_votes):
        raise ValueError("rigid-line object and plant vote classes must match")
    if parameters_by_class is not None:
        unknown_parameter_classes = set(parameters_by_class) - set(
            class_votes
        )
        if unknown_parameter_classes:
            raise ValueError(
                "rigid-line parameters reference unknown class: "
                f"{sorted(unknown_parameter_classes)[0]}"
            )

    rejected = np.zeros(len(coordinates), dtype=bool)
    class_reports: dict[str, dict[str, Any]] = {}
    for class_id, votes in class_votes.items():
        class_rejected, class_report = complete_railing_lines(
            coordinates,
            rgb=rgb,
            candidate_mask=candidate_mask,
            seed_mask=seed_mask,
            railing_votes=votes,
            plant_votes=class_plant_votes[class_id],
            support_height=support_height,
            parameters=(
                parameters_by_class.get(class_id, parameters)
                if parameters_by_class is not None
                else parameters
            ),
        )
        rejected |= class_rejected
        class_reports[class_id] = class_report
    return rejected, {
        "schema_version": 1,
        "strategy": "planned-rigid-line-classes-v1",
        "class_count": len(class_reports),
        "completed_point_count": int(rejected.sum()),
        "classes": class_reports,
    }


def _complete_rigid_component(
    coordinates: np.ndarray,
    *,
    rgb: np.ndarray,
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    object_votes: np.ndarray,
    plant_votes: np.ndarray,
    support_height: float,
    parameters: RigidComponentCompletionParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    if parameters.voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if parameters.minimum_seed_height_above_support < 0:
        raise ValueError(
            "minimum_seed_height_above_support must be non-negative"
        )
    if parameters.minimum_completion_height_above_support < 0:
        raise ValueError(
            "minimum_completion_height_above_support must be non-negative"
        )

    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    saturation = (rgb.max(axis=1) - rgb.min(axis=1)) / np.maximum(
        rgb.max(axis=1),
        1.0,
    )
    confirmed = (
        seed_mask
        & (object_votes >= parameters.minimum_object_votes)
        & (object_votes > plant_votes)
    )
    structural = (
        confirmed
        & (excess_green <= parameters.seed_excess_green_max)
        & (saturation <= parameters.seed_saturation_max)
    )
    elevated_seeds = structural & (
        coordinates[:, 2]
        >= support_height + parameters.minimum_seed_height_above_support
    )
    rejected = np.zeros(len(coordinates), dtype=bool)
    if not elevated_seeds.any():
        return rejected, {
            "schema_version": 1,
            "parameters": asdict(parameters),
            "confirmed_seed_count": int(confirmed.sum()),
            "structural_seed_count": int(structural.sum()),
            "elevated_seed_count": 0,
            "eligible_point_count": 0,
            "completed_point_count": 0,
            "status": "no_elevated_seeds",
        }

    seed_points = coordinates[elevated_seeds]
    lower = seed_points.min(axis=0) - parameters.bounds_margin
    upper = seed_points.max(axis=0) + parameters.bounds_margin
    lower[2] = (
        support_height
        + parameters.minimum_completion_height_above_support
    )
    strong_plant = plant_votes.astype(np.int16) >= (
        object_votes.astype(np.int16) + parameters.strong_plant_margin
    )
    eligible = (
        candidate_mask
        & (
            coordinates[:, 2]
            >= support_height
            + parameters.minimum_completion_height_above_support
        )
        & np.all(coordinates >= lower, axis=1)
        & np.all(coordinates <= upper, axis=1)
        & (excess_green <= parameters.completion_excess_green_max)
        & (saturation <= parameters.completion_saturation_max)
        & ~strong_plant
    )
    eligible |= elevated_seeds
    rows = np.flatnonzero(eligible)
    selected = coordinates[rows]
    grid_lower = selected.min(axis=0)
    voxels = np.floor(
        (selected - grid_lower) / parameters.voxel_size
    ).astype(np.int32)
    shape = tuple((voxels.max(axis=0) + 1).tolist())
    grid_voxels = int(np.prod(shape, dtype=np.int64))
    if grid_voxels > parameters.maximum_grid_voxels:
        raise ValueError(
            f"rigid component grid would contain {grid_voxels:,} cells"
        )
    occupancy = np.zeros(shape, dtype=bool)
    occupancy[tuple(voxels.T)] = True
    labels, component_count = ndimage.label(
        occupancy,
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    point_labels = labels[tuple(voxels.T)]
    seeded_labels = np.unique(point_labels[elevated_seeds[rows]])
    seeded_labels = seeded_labels[seeded_labels != 0]
    grown = np.isin(point_labels, seeded_labels)
    rejected[rows[grown]] = True
    return rejected, {
        "schema_version": 1,
        "parameters": asdict(parameters),
        "confirmed_seed_count": int(confirmed.sum()),
        "structural_seed_count": int(structural.sum()),
        "elevated_seed_count": int(elevated_seeds.sum()),
        "eligible_point_count": int(eligible.sum()),
        "voxel_grid_shape": list(shape),
        "occupied_voxel_count": int(occupancy.sum()),
        "component_count": int(component_count),
        "seeded_component_count": int(len(seeded_labels)),
        "completed_point_count": int(rejected.sum()),
        "status": "complete",
    }


def complete_rigid_component_classes(
    coordinates: np.ndarray,
    *,
    rgb: np.ndarray,
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    class_votes: Mapping[str, np.ndarray],
    class_plant_votes: Mapping[str, np.ndarray],
    support_height: float,
    parameters_by_class: Mapping[
        str, RigidComponentCompletionParameters
    ] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not class_votes:
        raise ValueError("at least one rigid component class is required")
    if set(class_votes) != set(class_plant_votes):
        raise ValueError(
            "rigid component object and plant vote classes must match"
        )
    unknown_parameters = set(parameters_by_class or {}) - set(class_votes)
    if unknown_parameters:
        raise ValueError(
            "rigid component parameters reference unknown class: "
            f"{sorted(unknown_parameters)[0]}"
        )

    rejected = np.zeros(len(coordinates), dtype=bool)
    reports: dict[str, dict[str, Any]] = {}
    for class_id, votes in class_votes.items():
        class_rejected, report = _complete_rigid_component(
            coordinates,
            rgb=rgb,
            candidate_mask=candidate_mask,
            seed_mask=seed_mask,
            object_votes=votes,
            plant_votes=class_plant_votes[class_id],
            support_height=support_height,
            parameters=(parameters_by_class or {}).get(
                class_id,
                RigidComponentCompletionParameters(),
            ),
        )
        rejected |= class_rejected
        reports[class_id] = report
    return rejected, {
        "schema_version": 1,
        "strategy": "source-verified-rigid-components-v1",
        "class_count": len(reports),
        "completed_point_count": int(rejected.sum()),
        "classes": reports,
    }


def _complete_rigid_surface(
    coordinates: np.ndarray,
    *,
    rgb: np.ndarray,
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    object_votes: np.ndarray,
    plant_votes: np.ndarray,
    parameters: RigidSurfaceCompletionParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    point_count = len(coordinates)
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    saturation = (rgb.max(axis=1) - rgb.min(axis=1)) / np.maximum(
        rgb.max(axis=1),
        1.0,
    )
    confirmed = (
        seed_mask
        & (object_votes >= parameters.minimum_object_votes)
        & (object_votes > plant_votes)
    )
    structural_seeds = confirmed & (
        (excess_green <= parameters.seed_excess_green_max)
        | (saturation <= parameters.seed_saturation_max)
    )
    seed_points = coordinates[structural_seeds]
    rejected = np.zeros(point_count, dtype=bool)
    surface_reports: list[dict[str, Any]] = []

    if len(seed_points) >= parameters.minimum_surface_seed_points:
        fit_points = seed_points
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

        centered = fit_points - center
        covariance = centered.T @ centered / len(fit_points)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)
        normal = vectors[:, order[0]]
        axes = vectors[:, order[1:]]
        seed_projection = (fit_points - center) @ axes
        bounds = np.quantile(seed_projection, (0.005, 0.995), axis=0)
        spans = bounds[1] - bounds[0]
        fit_distances = np.abs((fit_points - center) @ normal)
        planar = (
            float(np.quantile(fit_distances, 0.95))
            <= parameters.plane_distance
        )
        if planar and float(spans.max()) >= parameters.minimum_surface_span:
            projected = (coordinates - center) @ axes
            inside = np.all(
                (projected >= bounds[0] - parameters.bounds_margin)
                & (projected <= bounds[1] + parameters.bounds_margin),
                axis=1,
            )
            distance = np.abs((coordinates - center) @ normal)
            color_eligible = (
                excess_green <= parameters.completion_excess_green_max
            ) | (saturation <= parameters.completion_saturation_max)
            strong_plant = plant_votes.astype(np.int16) >= (
                object_votes.astype(np.int16)
                + parameters.strong_plant_margin
            )
            rejected = (
                candidate_mask
                & inside
                & (distance <= parameters.plane_distance)
                & color_eligible
                & ~strong_plant
            )
            surface_reports.append(
                {
                    "center": center.tolist(),
                    "normal": normal.tolist(),
                    "spans": spans.tolist(),
                    "seed_point_count": int(len(seed_points)),
                    "fit_point_count": int(len(fit_points)),
                    "seed_distance_95": float(
                        np.quantile(fit_distances, 0.95)
                    ),
                    "completed_point_count": int(rejected.sum()),
                }
            )

    return rejected, {
        "schema_version": 1,
        "parameters": asdict(parameters),
        "candidate_point_count": int(candidate_mask.sum()),
        "seed_candidate_point_count": int(seed_mask.sum()),
        "confirmed_seed_count": int(confirmed.sum()),
        "structural_seed_count": int(structural_seeds.sum()),
        "accepted_surface_count": len(surface_reports),
        "completed_point_count": int(rejected.sum()),
        "surfaces": surface_reports,
    }


def complete_rigid_surface_classes(
    coordinates: np.ndarray,
    *,
    rgb: np.ndarray,
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray | None = None,
    class_votes: Mapping[str, np.ndarray],
    class_plant_votes: Mapping[str, np.ndarray],
    parameters: RigidSurfaceCompletionParameters | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Complete planned rigid surfaces from class-exclusive semantic evidence."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (point_count, 3)")
    point_count = len(coordinates)
    if rgb.shape != (point_count, 3):
        raise ValueError(f"rgb must have shape ({point_count}, 3)")
    candidate_mask = _validate_vector(
        "candidate_mask",
        candidate_mask,
        point_count,
    ).astype(bool, copy=False)
    seed_mask = (
        candidate_mask
        if seed_mask is None
        else _validate_vector("seed_mask", seed_mask, point_count).astype(
            bool,
            copy=False,
        )
    )
    if not class_votes:
        raise ValueError("at least one rigid-surface class is required")
    if set(class_votes) != set(class_plant_votes):
        raise ValueError("rigid-surface object and plant vote classes must match")
    parameters = parameters or RigidSurfaceCompletionParameters()
    if parameters.minimum_surface_seed_points < 3:
        raise ValueError("minimum_surface_seed_points must be at least three")
    if not 0.0 < parameters.fit_trim_quantile <= 1.0:
        raise ValueError("fit_trim_quantile must be in (0, 1]")

    rejected = np.zeros(point_count, dtype=bool)
    class_reports: dict[str, dict[str, Any]] = {}
    for class_id, votes in class_votes.items():
        class_rejected, class_report = _complete_rigid_surface(
            coordinates,
            rgb=rgb,
            candidate_mask=candidate_mask,
            seed_mask=seed_mask,
            object_votes=_validate_vector(
                f"{class_id} object votes",
                votes,
                point_count,
            ),
            plant_votes=_validate_vector(
                f"{class_id} plant votes",
                class_plant_votes[class_id],
                point_count,
            ),
            parameters=parameters,
        )
        rejected |= class_rejected
        class_reports[class_id] = class_report
    return rejected, {
        "schema_version": 1,
        "strategy": "planned-rigid-surface-classes-v1",
        "class_count": len(class_reports),
        "completed_point_count": int(rejected.sum()),
        "classes": class_reports,
    }
