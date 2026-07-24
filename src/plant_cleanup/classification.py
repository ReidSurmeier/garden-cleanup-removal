from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from scipy import ndimage


class Reason(IntEnum):
    PLANT_SEED = 1
    PLANT_CONNECTED = 2
    UNCERTAIN_NEIGHBOR = 3
    SUPPORT_OR_GROUND = 4
    NON_TARGET_COMPONENT = 5
    SPARSE_FRAGMENT = 6


REASON_NAMES = {reason.value: reason.name.lower() for reason in Reason}


@dataclass(frozen=True)
class ClassificationParameters:
    voxel_size: float = 0.2
    plant_z_min: float = -4.2
    plant_z_max: float = 6.0
    candidate_excess_green: float = 3.0
    candidate_saturation: float = 0.18
    seed_excess_green: float = 30.0
    seed_z_min: float = -3.5
    seed_z_max: float = 5.5
    min_component_seed_points: int = 10_000
    max_target_components: int = 3
    uncertain_dilation_voxels: int = 1
    focus_x_min: float | None = None
    focus_x_max: float | None = None
    focus_y_min: float | None = None
    focus_y_max: float | None = None
    preservation_margin: float = 0.0
    preservation_bridge_voxels: int = 0


@dataclass(frozen=True)
class ClassificationResult:
    reasons: np.ndarray
    keep_mask: np.ndarray
    report: dict[str, Any]


def _colors(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colors = rgb.astype(np.float64)
    excess_green = 2.0 * colors[:, 1] - colors[:, 0] - colors[:, 2]
    saturation = (colors.max(axis=1) - colors.min(axis=1)) / np.maximum(
        colors.max(axis=1), 1.0
    )
    return excess_green, saturation


def classify_points(
    coordinates: np.ndarray,
    rgb: np.ndarray,
    parameters: ClassificationParameters,
) -> ClassificationResult:
    if len(coordinates) != len(rgb) or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates and RGB arrays must contain matching 3D points")
    if len(coordinates) == 0:
        raise ValueError("cannot classify an empty point cloud")
    if parameters.voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    excess_green, saturation = _colors(rgb)
    vertical_band = (coordinates[:, 2] >= parameters.plant_z_min) & (
        coordinates[:, 2] <= parameters.plant_z_max
    )
    focus = np.ones(len(coordinates), dtype=bool)
    for axis, lower, upper in (
        (0, parameters.focus_x_min, parameters.focus_x_max),
        (1, parameters.focus_y_min, parameters.focus_y_max),
    ):
        if lower is not None:
            focus &= coordinates[:, axis] >= lower
        if upper is not None:
            focus &= coordinates[:, axis] <= upper
    target_band = vertical_band & focus
    candidate = target_band & (
        (excess_green >= parameters.candidate_excess_green)
        | (saturation >= parameters.candidate_saturation)
    )
    seed = (
        (coordinates[:, 2] >= parameters.seed_z_min)
        & (coordinates[:, 2] <= parameters.seed_z_max)
        & focus
        & (excess_green >= parameters.seed_excess_green)
    )

    origin = coordinates.min(axis=0)
    voxel_indices = np.floor(
        (coordinates - origin) / parameters.voxel_size
    ).astype(np.int32)
    shape = tuple((voxel_indices.max(axis=0) + 1).tolist())
    grid = np.zeros(shape, dtype=bool)
    candidate_voxels = np.unique(voxel_indices[candidate], axis=0)
    if len(candidate_voxels):
        grid[tuple(candidate_voxels.T)] = True
    components, component_count = ndimage.label(
        grid, structure=np.ones((3, 3, 3), dtype=bool)
    )
    point_components = components[tuple(voxel_indices.T)]
    seed_counts = np.bincount(
        point_components[seed & candidate], minlength=component_count + 1
    )
    ranked = [
        int(component)
        for component in np.argsort(seed_counts[1:])[::-1] + 1
        if seed_counts[component] >= parameters.min_component_seed_points
    ][: parameters.max_target_components]
    selected_core = np.isin(point_components, ranked) & candidate

    selected_grid = np.isin(components, ranked) if ranked else np.zeros_like(grid)
    preservation_focus = np.ones(len(coordinates), dtype=bool)
    for axis, lower, upper in (
        (0, parameters.focus_x_min, parameters.focus_x_max),
        (1, parameters.focus_y_min, parameters.focus_y_max),
    ):
        if lower is not None:
            preservation_focus &= coordinates[:, axis] >= (
                lower - parameters.preservation_margin
            )
        if upper is not None:
            preservation_focus &= coordinates[:, axis] <= (
                upper + parameters.preservation_margin
            )
    preservation_region = (
        (coordinates[:, 2] >= parameters.plant_z_min - parameters.preservation_margin)
        & (coordinates[:, 2] <= parameters.plant_z_max + parameters.preservation_margin)
        & preservation_focus
    )
    preservation_candidate = preservation_region & (
        (excess_green >= parameters.candidate_excess_green)
        | (saturation >= parameters.candidate_saturation)
    )
    preservation_grid = np.zeros(shape, dtype=bool)
    preservation_voxels = np.unique(voxel_indices[preservation_candidate], axis=0)
    if len(preservation_voxels):
        preservation_grid[tuple(preservation_voxels.T)] = True
    connected_grid = preservation_grid
    if parameters.preservation_bridge_voxels:
        connected_grid = ndimage.binary_dilation(
            preservation_grid,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=parameters.preservation_bridge_voxels,
        )
    preservation_components, preservation_component_count = ndimage.label(
        connected_grid, structure=np.ones((3, 3, 3), dtype=bool)
    )
    touched_components = np.unique(preservation_components[selected_grid])
    touched_components = touched_components[touched_components != 0]
    preservation_point_components = preservation_components[tuple(voxel_indices.T)]
    selected = preservation_candidate & np.isin(
        preservation_point_components, touched_components
    )

    extended_grid = np.zeros(shape, dtype=bool)
    extended_voxels = np.unique(voxel_indices[selected], axis=0)
    if len(extended_voxels):
        extended_grid[tuple(extended_voxels.T)] = True
    near_selected_grid = ndimage.binary_dilation(
        extended_grid,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=parameters.uncertain_dilation_voxels,
    )
    near_selected = preservation_region & near_selected_grid[tuple(voxel_indices.T)]

    reasons = np.full(len(coordinates), Reason.NON_TARGET_COMPONENT, dtype=np.uint8)
    reasons[coordinates[:, 2] < parameters.plant_z_min] = Reason.SUPPORT_OR_GROUND
    reasons[coordinates[:, 2] > parameters.plant_z_max] = Reason.SPARSE_FRAGMENT

    unselected_candidate = preservation_candidate & ~selected
    small_component = np.zeros(len(coordinates), dtype=bool)
    if preservation_component_count:
        component_points = np.bincount(
            preservation_point_components[preservation_candidate],
            minlength=preservation_component_count + 1,
        )
        small_component = unselected_candidate & (
            component_points[preservation_point_components]
            < parameters.min_component_seed_points
        )
    reasons[small_component] = Reason.SPARSE_FRAGMENT
    reasons[near_selected & ~selected] = Reason.UNCERTAIN_NEIGHBOR
    reasons[selected & ~selected_core] = Reason.PLANT_CONNECTED
    reasons[selected_core & ~seed] = Reason.PLANT_CONNECTED
    reasons[selected_core & seed] = Reason.PLANT_SEED

    keep_mask = np.isin(
        reasons,
        [Reason.PLANT_SEED, Reason.PLANT_CONNECTED, Reason.UNCERTAIN_NEIGHBOR],
    )
    values, counts = np.unique(reasons, return_counts=True)
    report = {
        "parameters": asdict(parameters),
        "grid_shape": list(shape),
        "candidate_point_count": int(candidate.sum()),
        "preservation_candidate_point_count": int(preservation_candidate.sum()),
        "preserved_beyond_core_point_count": int((selected & ~selected_core).sum()),
        "seed_point_count": int(seed.sum()),
        "component_count": int(component_count),
        "selected_component_count": len(ranked),
        "selected_components": [
            {"label": component, "seed_points": int(seed_counts[component])}
            for component in ranked
        ],
        "reason_counts": {
            REASON_NAMES[int(value)]: int(count)
            for value, count in zip(values, counts, strict=True)
        },
        "kept_point_count": int(keep_mask.sum()),
        "rejected_point_count": int((~keep_mask).sum()),
    }
    return ClassificationResult(reasons=reasons, keep_mask=keep_mask, report=report)
