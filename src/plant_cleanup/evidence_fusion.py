from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class RecoveryParameters:
    """Conservative policy for promoting uncertain plant structure."""

    connectivity_voxel_size: float = 0.2
    ground_height_max: float = 0.3
    ground_normal_min: float = 0.8

    def __post_init__(self) -> None:
        if self.connectivity_voxel_size <= 0:
            raise ValueError("connectivity_voxel_size must be positive")
        if self.ground_height_max < 0:
            raise ValueError("ground_height_max must be nonnegative")
        if not 0 <= self.ground_normal_min <= 1:
            raise ValueError("ground_normal_min must be between zero and one")


def _connected_to_certain_plant(
    coordinates: np.ndarray,
    eligible: np.ndarray,
    certain: np.ndarray,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    connected = np.zeros(len(coordinates), dtype=bool)
    if not eligible.any() or not certain.any():
        return connected, {"component_count": 0, "seeded_component_count": 0}

    rows = np.flatnonzero(eligible)
    selected = coordinates[rows]
    lower = selected.min(axis=0)
    voxels = np.floor((selected - lower) / voxel_size).astype(np.int32)
    shape = tuple((voxels.max(axis=0) + 1).tolist())
    grid_voxels = int(np.prod(shape, dtype=np.int64))
    if grid_voxels > 100_000_000:
        raise ValueError(
            f"recovery voxel grid would contain {grid_voxels:,} cells; "
            "increase connectivity_voxel_size"
        )
    occupancy = np.zeros(shape, dtype=bool)
    occupancy[tuple(voxels.T)] = True
    labels, component_count = ndimage.label(
        occupancy, structure=np.ones((3, 3, 3), dtype=bool)
    )
    point_labels = labels[tuple(voxels.T)]
    certain_labels = np.unique(point_labels[certain[rows]])
    certain_labels = certain_labels[certain_labels != 0]
    connected[rows] = np.isin(point_labels, certain_labels)
    return connected, {
        "voxel_grid_shape": list(shape),
        "occupied_voxels": int(occupancy.sum()),
        "component_count": int(component_count),
        "seeded_component_count": int(len(certain_labels)),
    }


def recover_uncertain_structure(
    coordinates: np.ndarray,
    normals: np.ndarray,
    decisions: np.ndarray,
    plant_votes: np.ndarray,
    planter_votes: np.ndarray,
    *,
    plane_height: np.ndarray,
    parameters: RecoveryParameters = RecoveryParameters(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Promote target-connected uncertainty unless ground evidence contradicts it."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    decisions = np.asarray(decisions, dtype=np.uint8)
    plant_votes = np.asarray(plant_votes)
    planter_votes = np.asarray(planter_votes)
    plane_height = np.asarray(plane_height, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must contain 3D points")
    arrays = (normals, decisions, plant_votes, planter_votes, plane_height)
    if any(len(array) != len(coordinates) for array in arrays):
        raise ValueError("recovery evidence must match cloud length")

    uncertain = decisions == 5
    certain = decisions == 1
    plant_supported = uncertain & (plant_votes > planter_votes)
    planter_supported = uncertain & (planter_votes > plant_votes)
    ground_like = (
        uncertain
        & (plane_height <= parameters.ground_height_max)
        & (np.abs(normals[:, 2]) >= parameters.ground_normal_min)
    )
    connectivity_eligible = (
        certain | (uncertain & ~ground_like & ~planter_supported)
    )
    connected, component_report = _connected_to_certain_plant(
        coordinates,
        connectivity_eligible,
        certain,
        voxel_size=parameters.connectivity_voxel_size,
    )
    recovered_from_geometry = (
        uncertain & connected & ~ground_like & ~planter_supported
    )
    recovered = plant_supported | recovered_from_geometry
    result = decisions.copy()
    result[recovered] = 6
    return result, {
        "parameters": asdict(parameters),
        "uncertain_input_points": int(uncertain.sum()),
        "connected_uncertain_points": int((uncertain & connected).sum()),
        "ground_like_uncertain_points": int(ground_like.sum()),
        "planter_supported_uncertain_points": int(planter_supported.sum()),
        "recovered_plant_evidence_points": int(plant_supported.sum()),
        "recovered_geometry_points": int(
            (recovered_from_geometry & ~plant_supported).sum()
        ),
        "recovered_total_points": int(recovered.sum()),
        "retained_uncertain_points": int((result == 5).sum()),
        "components": component_report,
    }
