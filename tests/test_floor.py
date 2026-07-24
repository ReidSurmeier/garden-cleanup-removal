from __future__ import annotations

import numpy as np

from railing_removal.floor import (
    FloorRemovalParameters,
    GroundSurfaceCompletionParameters,
    complete_ground_surface_classes,
    remove_uncertain_floor,
)


def test_uncertain_floor_seeds_remove_the_same_coplanar_candidate_surface() -> None:
    floor = np.array(
        [(x, y, 0.0) for x in range(5) for y in range(5)],
        dtype=np.float64,
    )
    plant = np.array([(2.0, 2.0, 2.0), (2.0, 2.0, 3.0)], dtype=np.float64)
    coordinates = np.vstack((floor, plant))
    normals = np.tile((0.0, 0.0, 1.0), (len(coordinates), 1))
    rgb = np.tile((100, 100, 100), (len(coordinates), 1))
    rgb[len(floor) :] = (40, 150, 35)
    decisions = np.ones(len(coordinates), dtype=np.uint8)
    decisions[: len(floor) : 2] = 5

    keep, report = remove_uncertain_floor(
        coordinates,
        normals=normals,
        rgb=rgb,
        decisions=decisions,
        plant_votes=np.zeros(len(coordinates), dtype=np.uint8),
        background_votes=np.zeros(len(coordinates), dtype=np.uint8),
        parameters=FloorRemovalParameters(
            voxel_size=1.1,
            minimum_component_points=3,
            component_fraction=0.0,
            plane_distance=0.1,
            bounds_margin=0.1,
        ),
    )

    assert not keep[: len(floor)].any()
    assert keep[len(floor) :].all()
    assert report["coplanar_points_removed_from_candidate"] == len(floor) // 2


def test_adaptive_floor_growth_removes_green_ground_but_protects_root_evidence(
) -> None:
    uncertain_floor = np.array(
        [
            (float(x), float(y), 0.0)
            for x in range(3)
            for y in range(2)
        ],
        dtype=np.float64,
    )
    connected_green_ground = np.array(
        [
            (float(x), float(y), 0.0)
            for x in range(3, 10)
            for y in range(2)
        ],
        dtype=np.float64,
    )
    protected_root = np.array([(5.0, 0.4, 0.02)], dtype=np.float64)
    raised_plant = np.array([(5.0, 0.0, 2.0)], dtype=np.float64)
    coordinates = np.vstack(
        (
            uncertain_floor,
            connected_green_ground,
            protected_root,
            raised_plant,
        )
    )
    normals = np.tile((0.0, 0.0, 1.0), (len(coordinates), 1))
    rgb = np.tile((40, 150, 35), (len(coordinates), 1))
    decisions = np.ones(len(coordinates), dtype=np.uint8)
    decisions[: len(uncertain_floor)] = 5
    plant_votes = np.zeros(len(coordinates), dtype=np.uint8)
    plant_votes[-2] = 3

    keep, report = remove_uncertain_floor(
        coordinates,
        normals=normals,
        rgb=rgb,
        decisions=decisions,
        plant_votes=plant_votes,
        background_votes=np.zeros(len(coordinates), dtype=np.uint8),
        parameters=FloorRemovalParameters(
            voxel_size=1.1,
            minimum_component_points=2,
            component_fraction=0.0,
            plane_distance=0.1,
            bounds_margin=0.1,
            grow_floor_components=True,
            strong_plant_margin=2,
        ),
    )

    assert not keep[: len(uncertain_floor) + len(connected_green_ground)].any()
    assert keep[-2:].all()
    assert report["grown_floor_point_count"] == len(coordinates) - 2


def test_post_semantic_floor_growth_removes_brown_model_mistake_without_roots_or_leaves(
) -> None:
    uncertain_floor = np.array(
        [
            (float(x), float(y), 0.0)
            for x in range(3)
            for y in range(2)
        ],
        dtype=np.float64,
    )
    brown_mulch = np.array(
        [
            (float(x), float(y), 0.0)
            for x in range(3, 10)
            for y in range(2)
        ],
        dtype=np.float64,
    )
    root = np.array([(5.0, 0.4, 0.02)], dtype=np.float64)
    low_leaf = np.array([(6.0, 0.4, 0.03)], dtype=np.float64)
    raised_plant = np.array([(5.0, 0.0, 2.0)], dtype=np.float64)
    coordinates = np.vstack(
        (uncertain_floor, brown_mulch, root, low_leaf, raised_plant)
    )
    normals = np.tile((0.0, 0.0, 1.0), (len(coordinates), 1))
    normals[-3] = (1.0, 0.0, 0.0)
    rgb = np.tile((110, 85, 65), (len(coordinates), 1))
    rgb[: len(uncertain_floor)] = (100, 100, 100)
    rgb[-2:] = (35, 150, 40)
    decisions = np.ones(len(coordinates), dtype=np.uint8)
    decisions[: len(uncertain_floor)] = 5
    candidate = np.ones(len(coordinates), dtype=bool)
    candidate[: len(uncertain_floor)] = False
    plant_votes = np.zeros(len(coordinates), dtype=np.uint8)
    plant_votes[len(uncertain_floor) :] = 4

    keep, report = remove_uncertain_floor(
        coordinates,
        normals=normals,
        rgb=rgb,
        decisions=decisions,
        plant_votes=plant_votes,
        background_votes=np.zeros(len(coordinates), dtype=np.uint8),
        candidate_mask=candidate,
        parameters=FloorRemovalParameters(
            voxel_size=1.1,
            minimum_component_points=2,
            component_fraction=0.0,
            plane_distance=0.1,
            bounds_margin=0.1,
            grow_floor_components=True,
            strong_plant_margin=2,
        ),
    )

    mulch = slice(
        len(uncertain_floor),
        len(uncertain_floor) + len(brown_mulch),
    )
    assert not keep[mulch].any()
    assert keep[-3:].all()
    assert report["grown_floor_point_count"] == (
        len(uncertain_floor) + len(brown_mulch)
    )


def test_raised_semantic_ground_plane_is_completed_without_roots_or_plants(
) -> None:
    ground = np.array(
        [
            (float(x), float(y), 1.25)
            for x in range(11)
            for y in range(7)
        ],
        dtype=np.float64,
    )
    root = np.array([(5.0, 3.0, 1.27)], dtype=np.float64)
    raised_plant = np.array([(5.0, 3.0, 3.0)], dtype=np.float64)
    coordinates = np.vstack((ground, root, raised_plant))
    normals = np.tile((0.0, 0.0, 1.0), (len(coordinates), 1))
    normals[-2] = (1.0, 0.0, 0.0)
    object_votes = np.zeros(len(coordinates), dtype=np.uint8)
    evidence_mask = np.zeros(len(coordinates), dtype=bool)
    evidence_mask[: len(ground) : 3] = True
    object_votes[evidence_mask] = 2
    plant_votes = np.zeros(len(coordinates), dtype=np.uint8)
    plant_votes[-2:] = 5
    candidate_mask = ~evidence_mask

    rejected, report = complete_ground_surface_classes(
        coordinates,
        normals=normals,
        candidate_mask=candidate_mask,
        seed_mask=evidence_mask,
        class_votes={"turf_ground": object_votes},
        class_plant_votes={"turf_ground": plant_votes},
        parameters=GroundSurfaceCompletionParameters(
            minimum_surface_seed_points=10,
            minimum_surface_span=1.0,
            plane_distance=0.08,
        ),
    )

    assert rejected[: len(ground)][candidate_mask[: len(ground)]].all()
    assert not rejected[-2:].any()
    assert (
        report["classes"]["turf_ground"]["accepted_surface_count"] == 1
    )
