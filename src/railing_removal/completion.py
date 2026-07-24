from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


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
    minimum_line_seed_points: int = 15
    minimum_line_length: float = 0.70
    max_lines: int = 40
    ransac_iterations: int = 3_000
    random_seed: int = 172629


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
    railing_votes: np.ndarray,
    plant_votes: np.ndarray,
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
    railing_votes = _validate_vector(
        "railing_votes", railing_votes, point_count
    )
    plant_votes = _validate_vector("plant_votes", plant_votes, point_count)
    if parameters.minimum_line_seed_points < 2:
        raise ValueError("minimum_line_seed_points must be at least two")

    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    saturation = (rgb.max(axis=1) - rgb.min(axis=1)) / np.maximum(
        rgb.max(axis=1),
        1.0,
    )
    confirmed = (
        candidate_mask
        & (railing_votes >= parameters.minimum_railing_votes)
        & (railing_votes > plant_votes)
    )
    structural_seeds = (
        confirmed
        & (excess_green <= parameters.seed_excess_green_max)
        & (saturation <= parameters.seed_saturation_max)
    )
    remaining_ids = np.flatnonzero(structural_seeds)
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
        "confirmed_seed_count": int(confirmed.sum()),
        "structural_seed_count": int(structural_seeds.sum()),
        "accepted_line_count": len(line_reports),
        "completed_point_count": int(rejected.sum()),
        "lines": line_reports,
    }
    return rejected, report
