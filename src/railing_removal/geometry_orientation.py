from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _unit_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (count, 3)")
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & (lengths > 1e-12)
    return vectors[valid] / lengths[valid, None], valid


def _canonical_axis(
    axis: np.ndarray,
    reference_up: np.ndarray | None,
) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if reference_up is not None:
        reference = np.asarray(reference_up, dtype=np.float64)
        reference /= np.linalg.norm(reference)
        if float(axis @ reference) < 0:
            axis = -axis
        return axis
    dominant = int(np.argmax(np.abs(axis)))
    return axis if axis[dominant] >= 0 else -axis


def estimate_axes_from_normal_pairs(
    left_normals: np.ndarray,
    right_normals: np.ndarray,
    *,
    reference_up: np.ndarray | None = None,
    minimum_normal_separation_degrees: float = 8.0,
    maximum_axis_agreement_degrees: float = 12.0,
    minimum_support: int = 100,
    maximum_candidates: int = 5,
    maximum_axes: int = 50_000,
    maximum_seeds: int = 512,
) -> dict[str, Any]:
    """Find rigid axes where neighboring surface normals intersect."""

    left = np.asarray(left_normals, dtype=np.float64)
    right = np.asarray(right_normals, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("normal pairs must have matching (count, 3) shapes")
    if (
        minimum_support < 2
        or maximum_candidates < 1
        or maximum_axes < minimum_support
        or maximum_seeds < 1
    ):
        raise ValueError("axis estimator limits are invalid")
    if not 0 < minimum_normal_separation_degrees < 90:
        raise ValueError("normal separation must be between 0 and 90")
    if not 0 < maximum_axis_agreement_degrees < 45:
        raise ValueError("axis agreement must be between 0 and 45")

    left_unit, left_valid = _unit_rows(left)
    right_unit, right_valid = _unit_rows(right)
    jointly_valid = left_valid & right_valid
    left_unit = left[jointly_valid]
    right_unit = right[jointly_valid]
    left_unit /= np.linalg.norm(left_unit, axis=1)[:, None]
    right_unit /= np.linalg.norm(right_unit, axis=1)[:, None]
    crosses = np.cross(left_unit, right_unit)
    cross_lengths = np.linalg.norm(crosses, axis=1)
    separated = cross_lengths >= np.sin(
        np.radians(minimum_normal_separation_degrees)
    )
    axes = crosses[separated] / cross_lengths[separated, None]
    raw_axis_count = len(axes)
    if raw_axis_count < minimum_support:
        return {
            "schema_version": 1,
            "status": "insufficient_evidence",
            "normal_pair_count": int(len(left)),
            "usable_axis_count": int(raw_axis_count),
            "sampled_axis_count": int(raw_axis_count),
            "candidates": [],
        }
    if raw_axis_count > maximum_axes:
        axis_indices = np.linspace(
            0,
            raw_axis_count - 1,
            maximum_axes,
        ).round().astype(int)
        axes = axes[axis_indices]

    if len(axes) > maximum_seeds:
        seed_indices = np.linspace(
            0,
            len(axes) - 1,
            maximum_seeds,
        ).round().astype(int)
        seeds = axes[seed_indices]
    else:
        seeds = axes
    cosine = float(np.cos(np.radians(maximum_axis_agreement_degrees)))
    raw_candidates: list[dict[str, Any]] = []
    for seed in seeds:
        mask = np.abs(axes @ seed) >= cosine
        support = int(np.count_nonzero(mask))
        if support < minimum_support:
            continue
        scatter = axes[mask].T @ axes[mask]
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        refined = _canonical_axis(eigenvectors[:, -1], reference_up)
        refined_mask = np.abs(axes @ refined) >= cosine
        refined_support = int(np.count_nonzero(refined_mask))
        residuals = np.degrees(
            np.arccos(
                np.clip(
                    np.abs(axes[refined_mask] @ refined),
                    0.0,
                    1.0,
                )
            )
        )
        raw_candidates.append(
            {
                "axis": refined.tolist(),
                "support_count": refined_support,
                "support_fraction": refined_support / len(axes),
                "median_residual_degrees": float(np.median(residuals)),
                "maximum_residual_degrees": float(np.max(residuals)),
                "eigenvalues": eigenvalues.tolist(),
            }
        )
    raw_candidates.sort(
        key=lambda item: (
            -item["support_count"],
            item["median_residual_degrees"],
        )
    )
    candidates: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        axis = np.asarray(candidate["axis"], dtype=np.float64)
        duplicate = any(
            abs(
                float(
                    axis
                    @ np.asarray(existing["axis"], dtype=np.float64)
                )
            )
            >= cosine
            for existing in candidates
        )
        if not duplicate:
            candidates.append(candidate)
        if len(candidates) == maximum_candidates:
            break
    return {
        "schema_version": 1,
        "status": "usable" if candidates else "insufficient_evidence",
        "normal_pair_count": int(len(left)),
        "usable_axis_count": int(raw_axis_count),
        "sampled_axis_count": int(len(axes)),
        "candidates": candidates,
    }


def estimate_rigid_axes_from_cloud(
    coordinates: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    *,
    reference_up: np.ndarray | None = None,
    maximum_sample_points: int = 60_000,
    neighbors: int = 20,
    maximum_vertical_surface_normal_dot: float = 0.75,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Sample neutral rigid surfaces and estimate their repeated 3D axes."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or normals.shape != coordinates.shape
        or colors.shape != coordinates.shape
    ):
        raise ValueError("cloud coordinates, normals, and colors must be Nx3")
    if not 0 < maximum_vertical_surface_normal_dot < 1:
        raise ValueError("vertical surface normal dot must be in (0, 1)")
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = (
        np.isfinite(coordinates).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & (normal_lengths > 0.5)
    )
    if reference_up is not None:
        reference = np.asarray(reference_up, dtype=np.float64)
        reference /= np.linalg.norm(reference)
        normal_up_dot = np.zeros(len(normals), dtype=np.float64)
        normal_up_dot[valid] = np.abs(
            (normals[valid] / normal_lengths[valid, None]) @ reference
        )
        valid &= normal_up_dot <= maximum_vertical_surface_normal_dot
    vertical_surface_point_count = int(np.count_nonzero(valid))
    scaled_colors = colors / 255.0
    chroma = np.max(scaled_colors, axis=1) - np.min(
        scaled_colors,
        axis=1,
    )
    luminance = np.mean(scaled_colors, axis=1)
    neutral = valid & (chroma <= 0.28) & (luminance >= 0.05) & (
        luminance <= 0.95
    )
    candidates = np.flatnonzero(neutral)
    selection_basis = "neutral_rigid_color"
    if len(candidates) < 1000:
        candidates = np.flatnonzero(valid)
        selection_basis = "all_valid_normals_fallback"
    rng = np.random.default_rng(random_seed)
    if len(candidates) > maximum_sample_points:
        candidates = np.sort(
            rng.choice(
                candidates,
                size=maximum_sample_points,
                replace=False,
            )
        )
    if len(candidates) < neighbors:
        return {
            "schema_version": 1,
            "status": "insufficient_evidence",
            "selection_basis": selection_basis,
            "vertical_surface_point_count": vertical_surface_point_count,
            "sample_point_count": int(len(candidates)),
            "candidates": [],
        }

    points = coordinates[candidates]
    selected_normals = normals[candidates]
    tree = cKDTree(points)
    _, neighbor_indices = tree.query(
        points,
        k=min(neighbors, len(points)),
        workers=-1,
    )
    left = np.repeat(
        selected_normals,
        neighbor_indices.shape[1] - 1,
        axis=0,
    )
    right = selected_normals[neighbor_indices[:, 1:].reshape(-1)]
    result = estimate_axes_from_normal_pairs(
        left,
        right,
        reference_up=reference_up,
        minimum_support=max(100, round(50_000 * 0.005)),
    )
    return {
        **result,
        "selection_basis": selection_basis,
        "source_point_count": int(len(coordinates)),
        "vertical_surface_point_count": vertical_surface_point_count,
        "sample_point_count": int(len(candidates)),
    }
