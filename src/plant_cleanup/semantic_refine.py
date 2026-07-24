from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from plant_cleanup.evidence_fusion import RecoveryParameters, recover_uncertain_structure
from plant_cleanup.geometry_cleanup import write_decision_cloud
from plant_cleanup.plyio import read_cloud


@dataclass(frozen=True)
class SemanticParameters:
    height_band: float = 3.0
    growth_radius: float = 0.6
    excess_green_max: float = 10.0
    min_planter_views: int = 1
    ground_normal_min: float = 0.8
    ground_seed_height_fraction: float = 0.35
    growth_height_multiplier: float = 3.0

    def __post_init__(self) -> None:
        if self.height_band <= 0 or self.growth_radius < 0:
            raise ValueError("height band must be positive and growth radius nonnegative")
        if self.min_planter_views < 1:
            raise ValueError("min_planter_views must be positive")
        if not 0 <= self.ground_normal_min <= 1:
            raise ValueError("ground normal threshold must be between zero and one")
        if not 0 < self.ground_seed_height_fraction <= 1:
            raise ValueError("ground seed height fraction must be in (0, 1]")
        if self.growth_height_multiplier < 1:
            raise ValueError("growth height multiplier must be at least one")


def _support_plane_height(
    coordinates: np.ndarray,
    normals: np.ndarray,
    excess_green: np.ndarray,
    *,
    support_height: float,
    parameters: SemanticParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure height from the observed support plane, including sloped ground."""
    normal_candidates = (
        (np.abs(normals[:, 2]) >= parameters.ground_normal_min)
        & (excess_green < parameters.excess_green_max)
        & (
            coordinates[:, 2]
            <= support_height
            + parameters.height_band * parameters.growth_height_multiplier
        )
    )
    if not normal_candidates.any():
        return coordinates[:, 2] - support_height, {
            "strategy": "horizontal_z_fallback",
            "coefficients": [0.0, 0.0, float(support_height)],
            "normal_candidate_points": 0,
            "offset_candidate_points": 0,
        }

    candidate_normals = np.asarray(normals[normal_candidates], dtype=np.float64)
    candidate_normals[candidate_normals[:, 2] < 0] *= -1
    plane_normal = np.median(candidate_normals, axis=0)
    length = float(np.linalg.norm(plane_normal))
    if not np.isfinite(length) or length < 1e-8 or abs(plane_normal[2]) < 1e-4:
        return coordinates[:, 2] - support_height, {
            "strategy": "horizontal_z_fallback",
            "coefficients": [0.0, 0.0, float(support_height)],
            "normal_candidate_points": int(normal_candidates.sum()),
            "offset_candidate_points": 0,
        }
    plane_normal /= length

    fit_band = parameters.height_band * 0.4
    offset_candidates = normal_candidates & (
        np.abs(coordinates[:, 2] - support_height) <= fit_band
    )
    if not offset_candidates.any():
        offset_candidates = normal_candidates
    offset = float(
        np.median(
            np.asarray(coordinates[offset_candidates], dtype=np.float64)
            @ plane_normal
        )
    )
    signed_distance = (
        np.asarray(coordinates, dtype=np.float64) @ plane_normal - offset
    ).astype(np.float32)
    coefficients = [
        float(-plane_normal[0] / plane_normal[2]),
        float(-plane_normal[1] / plane_normal[2]),
        float(offset / plane_normal[2]),
    ]
    return signed_distance, {
        "strategy": "median_low_support_normals",
        "coefficients": coefficients,
        "normal": plane_normal.tolist(),
        "normal_candidate_points": int(normal_candidates.sum()),
        "offset_candidate_points": int(offset_candidates.sum()),
    }


def ground_prompt_candidates(
    coordinates: np.ndarray,
    normals: np.ndarray,
    rgb: np.ndarray,
    *,
    support_height: float,
    support_cutoff: float,
    parameters: SemanticParameters = SemanticParameters(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify low, horizontal, non-green points relative to fitted support."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must contain 3D points")
    if normals.shape != coordinates.shape or rgb.shape != coordinates.shape:
        raise ValueError("ground prompt evidence must match 3D coordinates")
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    plane_height, plane_report = _support_plane_height(
        coordinates,
        normals,
        excess_green,
        support_height=float(support_height),
        parameters=parameters,
    )
    support_clearance = max(0.0, float(support_cutoff) - float(support_height))
    mask = (
        (
            plane_height
            <= support_clearance
            + parameters.height_band * parameters.ground_seed_height_fraction
        )
        & (np.abs(normals[:, 2]) >= parameters.ground_normal_min)
        & (excess_green < parameters.excess_green_max)
    )
    return mask, {
        "support_plane": plane_report,
        "candidate_points": int(mask.sum()),
        "support_clearance": support_clearance,
    }


def _seeded_component_mask(
    coordinates: np.ndarray,
    eligible: np.ndarray,
    seeds: np.ndarray,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.zeros(len(coordinates), dtype=bool)
    if voxel_size <= 0 or not eligible.any() or not seeds.any():
        result[seeds] = True
        return result, {"strategy": "seeds_only", "component_count": 0}

    rows = np.flatnonzero(eligible)
    selected = coordinates[rows]
    lower = selected.min(axis=0)
    voxels = np.floor((selected - lower) / voxel_size).astype(np.int32)
    shape = tuple((voxels.max(axis=0) + 1).tolist())
    grid_voxels = int(np.prod(shape, dtype=np.int64))
    if grid_voxels > 100_000_000:
        raise ValueError(
            f"semantic voxel grid would contain {grid_voxels:,} cells; "
            "increase growth_radius"
        )
    occupancy = np.zeros(shape, dtype=bool)
    occupancy[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    labels, component_count = ndimage.label(
        occupancy, structure=np.ones((3, 3, 3), dtype=bool)
    )
    point_labels = labels[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    seed_rows = np.flatnonzero(seeds[rows])
    seeded_labels = np.unique(point_labels[seed_rows])
    seeded_labels = seeded_labels[seeded_labels != 0]
    grown = np.isin(point_labels, seeded_labels)
    result[rows[grown]] = True
    return result, {
        "strategy": "seeded_voxel_components",
        "voxel_size": float(voxel_size),
        "voxel_grid_shape": list(shape),
        "occupied_voxels": int(occupancy.sum()),
        "component_count": int(component_count),
        "seeded_component_count": int(len(seeded_labels)),
    }


def refine_with_semantics(
    source_path: Path,
    geometry_decisions: np.ndarray,
    plant_votes: np.ndarray,
    planter_votes: np.ndarray,
    *,
    support_height: float | None = None,
    support_cutoff: float,
    output_dir: Path,
    parameters: SemanticParameters = SemanticParameters(),
    background_class_votes: Mapping[str, np.ndarray] | None = None,
    background_class_plant_votes: Mapping[str, np.ndarray] | None = None,
    background_class_policies: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Grow model-backed planter evidence only through low, non-green geometry."""
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    cloud = read_cloud(source_path)
    arrays = (geometry_decisions, plant_votes, planter_votes)
    if any(len(array) != len(cloud) for array in arrays):
        raise ValueError("semantic votes and geometry decisions must match cloud length")
    scene_votes = {
        class_id: np.asarray(votes, dtype=np.uint8)
        for class_id, votes in (background_class_votes or {}).items()
    }
    if any(len(votes) != len(cloud) for votes in scene_votes.values()):
        raise ValueError("scene class votes must match cloud length")
    scene_plant_votes = {
        class_id: np.asarray(votes, dtype=np.uint8)
        for class_id, votes in (background_class_plant_votes or {}).items()
    }
    if set(scene_plant_votes) - set(scene_votes):
        raise ValueError("paired scene plant votes have no matching object class")
    if any(len(votes) != len(cloud) for votes in scene_plant_votes.values()):
        raise ValueError("paired scene plant votes must match cloud length")
    class_policies = dict(background_class_policies or {})
    unknown_policy_classes = set(class_policies) - set(scene_votes)
    if unknown_policy_classes:
        raise ValueError(
            "scene class policies have no matching votes: "
            f"{sorted(unknown_policy_classes)}"
        )
    invalid_policies = {
        class_id: policy
        for class_id, policy in class_policies.items()
        if policy
        not in {
            "strict_background",
            "ground_surface",
            "class_exclusive_background",
        }
    }
    if invalid_policies:
        raise ValueError(f"unsupported scene class policies: {invalid_policies}")

    coordinates = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(
        np.float32, copy=False
    )
    rgb = np.column_stack((cloud["red"], cloud["green"], cloud["blue"])).astype(
        np.float32
    )
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    support_height = (
        float(support_cutoff) if support_height is None else float(support_height)
    )
    plane_height, plane_report = _support_plane_height(
        coordinates,
        np.column_stack((cloud["nx"], cloud["ny"], cloud["nz"])),
        excess_green,
        support_height=support_height,
        parameters=parameters,
    )
    support_clearance = max(0.0, float(support_cutoff) - support_height)
    plant_candidate = geometry_decisions == 1
    review_candidate = np.isin(geometry_decisions, [1, 5])
    normal_z = np.abs(np.asarray(cloud["nz"], dtype=np.float32))
    # Preserve the pre-plan behavior when no explicit policies were supplied.
    effective_policies = {
        class_id: class_policies.get(
            class_id,
            (
                "strict_background"
                if class_id == "railing"
                else "ground_surface"
                if class_id == "terrain"
                else ""
            ),
        )
        for class_id in scene_votes
    }
    scene_class_background: dict[str, np.ndarray] = {}
    for class_id, votes in scene_votes.items():
        policy = effective_policies[class_id]
        if policy == "strict_background":
            class_background = (
                (votes >= parameters.min_planter_views)
                & (votes > plant_votes)
            )
        elif policy == "class_exclusive_background":
            paired_plant = scene_plant_votes.get(class_id, plant_votes)
            class_background = (
                (votes >= parameters.min_planter_views)
                & (votes > paired_plant)
            )
        elif policy == "ground_surface":
            class_background = (
                (votes >= parameters.min_planter_views)
                & (votes >= plant_votes)
                & (normal_z >= parameters.ground_normal_min)
            )
        else:
            class_background = np.zeros(len(cloud), dtype=bool)
        scene_class_background[class_id] = review_candidate & class_background
    scene_background = (
        np.logical_or.reduce(list(scene_class_background.values()))
        if scene_class_background
        else np.zeros(len(cloud), dtype=bool)
    )
    ground_surface_seeds = (
        np.logical_or.reduce(
            [
                scene_class_background[class_id]
                for class_id, policy in effective_policies.items()
                if policy == "ground_surface"
            ]
        )
        if any(policy == "ground_surface" for policy in effective_policies.values())
        else np.zeros(len(cloud), dtype=bool)
    )
    low_nongreen = (
        (
            plane_height
            < support_clearance
            + parameters.height_band * parameters.growth_height_multiplier
        )
        & (excess_green < parameters.excess_green_max)
    )
    semantic_seeds = (
        (
            plant_candidate
            & low_nongreen
            & (planter_votes >= parameters.min_planter_views)
            & (planter_votes > plant_votes)
        )
        | scene_background
    )
    plant_protected = (
        plant_candidate
        & (plant_votes > planter_votes)
        & ~scene_background
    )
    ground_height_band = (
        plane_height
        < support_clearance
        + parameters.height_band * parameters.ground_seed_height_fraction
    )
    ground_seeds = (
        plant_candidate
        & ground_height_band
        & (normal_z >= parameters.ground_normal_min)
        & ~plant_protected
    )
    horizontal_ground_corridor = (
        low_nongreen
        & ground_height_band
        & (
            normal_z >= parameters.ground_normal_min
        )
    )
    planter_supported_corridor = (
        low_nongreen
        & (planter_votes >= parameters.min_planter_views)
        & (planter_votes > plant_votes)
    )
    semantic_eligible = (
        ((plant_candidate & planter_supported_corridor) | scene_background)
        & ~plant_protected
    )
    semantic_reject, growth_report = _seeded_component_mask(
        coordinates,
        semantic_eligible,
        semantic_seeds,
        voxel_size=parameters.growth_radius,
    )
    ground_surface_reject, ground_surface_completion_report = (
        _seeded_component_mask(
            coordinates,
            (
                review_candidate
                & (normal_z >= parameters.ground_normal_min)
            ),
            ground_surface_seeds,
            voxel_size=parameters.growth_radius,
        )
    )
    semantic_reject |= ground_surface_reject
    ground_uncertain, ground_growth_report = _seeded_component_mask(
        coordinates,
        plant_candidate & horizontal_ground_corridor & ~plant_protected,
        ground_seeds,
        voxel_size=parameters.growth_radius,
    )

    decisions = np.asarray(geometry_decisions, dtype=np.uint8).copy()
    decisions[ground_uncertain] = 5
    decisions[semantic_reject] = 4
    decisions, recovery_report = recover_uncertain_structure(
        coordinates,
        np.column_stack((cloud["nx"], cloud["ny"], cloud["nz"])),
        decisions,
        plant_votes,
        planter_votes,
        plane_height=plane_height,
        parameters=RecoveryParameters(
            connectivity_voxel_size=parameters.growth_radius,
            ground_height_max=(
                support_clearance
                + parameters.height_band * parameters.ground_seed_height_fraction
            ),
            ground_normal_min=parameters.ground_normal_min,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plant_path = output_dir / "plant-semantic.ply"
    certain_path = output_dir / "plant-certain-semantic.ply"
    recovered_path = output_dir / "recovered-semantic.ply"
    uncertain_path = output_dir / "uncertain-semantic.ply"
    review_retained_path = output_dir / "review-retained-semantic.ply"
    rejected_path = output_dir / "rejected-semantic.ply"
    decisions_path = output_dir / "decision-codes.npy"
    plant = np.isin(decisions, [1, 6])
    write_decision_cloud(cloud, plant_path, plant, decisions)
    write_decision_cloud(cloud, certain_path, decisions == 1, decisions)
    write_decision_cloud(cloud, recovered_path, decisions == 6, decisions)
    write_decision_cloud(cloud, uncertain_path, decisions == 5, decisions)
    retained = np.isin(decisions, [1, 5, 6])
    write_decision_cloud(cloud, review_retained_path, retained, decisions)
    write_decision_cloud(cloud, rejected_path, ~retained, decisions)
    np.save(decisions_path, decisions)

    counts = {
        "plant": int(plant.sum()),
        "certain_plant": int(np.count_nonzero(decisions == 1)),
        "recovered_plant": int(np.count_nonzero(decisions == 6)),
        "rejected_support": int(np.count_nonzero(decisions == 2)),
        "rejected_incoherent": int(np.count_nonzero(decisions == 3)),
        "rejected_semantic_background": int(np.count_nonzero(decisions == 4)),
        "uncertain": int(np.count_nonzero(decisions == 5)),
    }
    return {
        "source": str(source_path),
        "plant": str(plant_path),
        "certain_plant": str(certain_path),
        "recovered_plant": str(recovered_path),
        "uncertain": str(uncertain_path),
        "review_retained": str(review_retained_path),
        "rejected": str(rejected_path),
        "decisions": str(decisions_path),
        "support_cutoff": float(support_cutoff),
        "support_height": support_height,
        "support_plane": plane_report,
        "semantic_growth": growth_report,
        "ground_uncertainty_growth": ground_growth_report,
        "ground_surface_completion": {
            **ground_surface_completion_report,
            "seed_points": int(ground_surface_seeds.sum()),
            "completed_points": int(ground_surface_reject.sum()),
        },
        "uncertain_recovery": recovery_report,
        "semantic_seed_points": int((semantic_seeds | ground_seeds).sum()),
        "model_semantic_seed_points": int(semantic_seeds.sum()),
        "horizontal_ground_seed_points": int(ground_seeds.sum()),
        "geometry_ground_uncertain_points": int(ground_uncertain.sum()),
        "plant_evidence_protected_points": int(plant_protected.sum()),
        "scene_class_seed_points": {
            class_id: int(mask.sum())
            for class_id, mask in scene_class_background.items()
        },
        "scene_class_policies": effective_policies,
        "parameters": asdict(parameters),
        "counts": counts,
        "decision_codes": {
            "1": "plant",
            "2": "rejected_support",
            "3": "rejected_incoherent",
            "4": "rejected_semantic_background",
            "5": "uncertain",
            "6": "recovered_plant",
        },
    }
