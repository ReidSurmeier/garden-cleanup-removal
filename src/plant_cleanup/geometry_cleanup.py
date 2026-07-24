from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from plant_cleanup.plyio import VERTEX_DTYPE, read_cloud


PLANT = np.uint8(1)
REJECTED_SUPPORT = np.uint8(2)
REJECTED_INCOHERENT = np.uint8(3)


@dataclass(frozen=True)
class CleanupParameters:
    voxel_size: float = 0.2
    support_bin_size: float = 0.25
    support_clearance: float = 1.0
    support_normal_min: float = 0.8
    vegetation_excess_green_min: float = 15.0
    min_component_points: int = 500

    def __post_init__(self) -> None:
        if self.voxel_size <= 0 or self.support_bin_size <= 0:
            raise ValueError("voxel and support bin sizes must be positive")
        if self.support_clearance < 0 or self.min_component_points < 1:
            raise ValueError("clearance must be nonnegative and component size positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinates(cloud: np.ndarray) -> np.ndarray:
    return np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(
        np.float32, copy=False
    )


def _support_height(cloud: np.ndarray, parameters: CleanupParameters) -> float:
    rgb = np.column_stack((cloud["red"], cloud["green"], cloud["blue"])).astype(
        np.float32
    )
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    candidates = (
        (np.abs(cloud["nz"]) >= parameters.support_normal_min)
        & (excess_green < parameters.vegetation_excess_green_min)
    )
    heights = np.asarray(cloud["z"])[candidates]
    if not len(heights):
        raise ValueError("no low-vegetation horizontal support points were found")
    origin = float(heights.min())
    bins = np.floor((heights - origin) / parameters.support_bin_size).astype(
        np.int64
    )
    winning_bin = int(np.argmax(np.bincount(bins)))
    winning_heights = heights[bins == winning_bin]
    return float(np.median(winning_heights))


def _coherent_mask(
    coordinates: np.ndarray,
    candidate_mask: np.ndarray,
    parameters: CleanupParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = coordinates[candidate_mask]
    if not len(candidates):
        raise ValueError("support cutoff removed every point")
    lower = candidates.min(axis=0)
    voxels = np.floor((candidates - lower) / parameters.voxel_size).astype(np.int32)
    shape = tuple((voxels.max(axis=0) + 1).tolist())
    grid_voxels = int(np.prod(shape, dtype=np.int64))
    if grid_voxels > 100_000_000:
        raise ValueError(
            f"voxel grid would contain {grid_voxels:,} cells; increase voxel_size"
        )
    occupancy = np.zeros(shape, dtype=bool)
    occupancy[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    labels, component_count = ndimage.label(
        occupancy, structure=np.ones((3, 3, 3), dtype=bool)
    )
    point_labels = labels[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    point_counts = np.bincount(point_labels)
    coherent_labels = np.flatnonzero(point_counts >= parameters.min_component_points)
    coherent_labels = coherent_labels[coherent_labels != 0]
    candidate_keep = np.isin(point_labels, coherent_labels)
    result = np.zeros(len(coordinates), dtype=bool)
    result[np.flatnonzero(candidate_mask)] = candidate_keep
    largest = sorted(
        (int(count) for count in point_counts[1:]), reverse=True
    )[:10]
    return result, {
        "voxel_grid_shape": list(shape),
        "occupied_voxels": int(occupancy.sum()),
        "component_count": int(component_count),
        "coherent_component_count": int(len(coherent_labels)),
        "largest_component_point_counts": largest,
    }


def write_decision_cloud(
    source: np.ndarray,
    output: Path,
    mask: np.ndarray,
    decision_codes: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = int(mask.sum())
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment classification is the plant-clean decision code\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar classification\n"
        "property uint source_index\n"
        "end_header\n"
    ).encode("ascii")
    with output.open("wb") as destination:
        destination.write(header)
        for start in range(0, len(source), 1_000_000):
            stop = min(start + 1_000_000, len(source))
            local_mask = mask[start:stop]
            if not local_mask.any():
                continue
            selected = np.array(source[start:stop][local_mask], dtype=VERTEX_DTYPE)
            selected["classification"] = decision_codes[start:stop][local_mask]
            destination.write(selected.tobytes())


def run_geometry_cleanup(
    source_path: Path,
    plant_path: Path,
    rejected_path: Path,
    decisions_path: Path,
    parameters: CleanupParameters = CleanupParameters(),
) -> dict[str, Any]:
    """Classify support structures and detached fragments without changing source data."""
    source_path = source_path.resolve()
    plant_path = plant_path.resolve()
    rejected_path = rejected_path.resolve()
    decisions_path = decisions_path.resolve()
    cloud = read_cloud(source_path)
    coordinates = _coordinates(cloud)
    support_height = _support_height(cloud, parameters)
    above_support = coordinates[:, 2] > support_height + parameters.support_clearance
    coherent, component_report = _coherent_mask(
        coordinates, above_support, parameters
    )

    decision_codes = np.full(len(cloud), REJECTED_SUPPORT, dtype=np.uint8)
    decision_codes[above_support] = REJECTED_INCOHERENT
    decision_codes[coherent] = PLANT
    rejected = ~coherent
    write_decision_cloud(cloud, plant_path, coherent, decision_codes)
    write_decision_cloud(cloud, rejected_path, rejected, decision_codes)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(decisions_path, decision_codes)

    counts = {
        "source": int(len(cloud)),
        "plant": int(np.count_nonzero(decision_codes == PLANT)),
        "rejected_support": int(
            np.count_nonzero(decision_codes == REJECTED_SUPPORT)
        ),
        "rejected_incoherent": int(
            np.count_nonzero(decision_codes == REJECTED_INCOHERENT)
        ),
    }
    return {
        "source": str(source_path),
        "plant": str(plant_path),
        "rejected": str(rejected_path),
        "decisions": str(decisions_path),
        "source_sha256": _sha256(source_path),
        "plant_sha256": _sha256(plant_path),
        "rejected_sha256": _sha256(rejected_path),
        "support_height": support_height,
        "support_cutoff": support_height + parameters.support_clearance,
        "parameters": asdict(parameters),
        "counts": counts,
        "components": component_report,
        "decision_codes": {
            "1": "plant",
            "2": "rejected_support",
            "3": "rejected_incoherent",
        },
    }
