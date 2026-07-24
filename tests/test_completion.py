from __future__ import annotations

import numpy as np

from railing_removal import LineCompletionParameters, complete_railing_lines
from railing_removal.completion import (
    RigidComponentCompletionParameters,
    RigidSurfaceCompletionParameters,
    complete_rigid_component_classes,
    complete_rigid_line_classes,
    complete_rigid_surface_classes,
)


def test_user_can_complete_an_occluded_rail_from_sparse_semantic_seeds() -> None:
    rail = np.array(
        [(x, 0.0, 1.0) for x in np.linspace(-5.0, 5.0, 101)],
        dtype=np.float64,
    )
    plant = np.array(
        [(-1.0, 2.0, 0.5), (0.0, 2.0, 1.5), (1.0, 2.0, 2.5)],
        dtype=np.float64,
    )
    coordinates = np.vstack((rail, plant))
    rgb = np.full((len(coordinates), 3), 90, dtype=np.uint8)
    rgb[len(rail) :] = (40, 150, 35)
    railing_votes = np.zeros(len(coordinates), dtype=np.uint8)
    railing_votes[: len(rail) : 5] = 2

    rejected, report = complete_railing_lines(
        coordinates,
        rgb=rgb,
        candidate_mask=np.ones(len(coordinates), dtype=bool),
        railing_votes=railing_votes,
        plant_votes=np.zeros(len(coordinates), dtype=np.uint8),
        parameters=LineCompletionParameters(
            seed_distance=0.03,
            completion_radius=0.08,
            minimum_line_seed_points=5,
            minimum_line_length=1.0,
            ransac_iterations=500,
            random_seed=11,
        ),
    )

    assert rejected[: len(rail)].all()
    assert not rejected[len(rail) :].any()
    assert report["accepted_line_count"] == 1
    assert report["completed_point_count"] == len(rail)


def test_green_planar_false_positive_does_not_become_a_railing() -> None:
    coordinates = np.array(
        [
            (x, y, 0.0)
            for x in np.linspace(-2.0, 2.0, 9)
            for y in np.linspace(-2.0, 2.0, 9)
        ],
        dtype=np.float64,
    )
    rgb = np.tile(np.array((45, 150, 35), dtype=np.uint8), (len(coordinates), 1))
    railing_votes = np.full(len(coordinates), 2, dtype=np.uint8)

    rejected, report = complete_railing_lines(
        coordinates,
        rgb=rgb,
        candidate_mask=np.ones(len(coordinates), dtype=bool),
        railing_votes=railing_votes,
        plant_votes=np.zeros(len(coordinates), dtype=np.uint8),
        parameters=LineCompletionParameters(
            minimum_line_seed_points=5,
            ransac_iterations=200,
        ),
    )

    assert not rejected.any()
    assert report["confirmed_seed_count"] == len(coordinates)
    assert report["structural_seed_count"] == 0
    assert report["accepted_line_count"] == 0


def test_strong_plant_evidence_protects_a_point_crossing_an_accepted_rail() -> None:
    coordinates = np.array(
        [(x, 0.0, 0.0) for x in np.linspace(-2.0, 2.0, 41)],
        dtype=np.float64,
    )
    rgb = np.full((len(coordinates), 3), 100, dtype=np.uint8)
    railing_votes = np.zeros(len(coordinates), dtype=np.uint8)
    railing_votes[::4] = 2
    protected_id = 20
    plant_votes = np.zeros(len(coordinates), dtype=np.uint8)
    plant_votes[protected_id] = 5

    rejected, _ = complete_railing_lines(
        coordinates,
        rgb=rgb,
        candidate_mask=np.ones(len(coordinates), dtype=bool),
        railing_votes=railing_votes,
        plant_votes=plant_votes,
        parameters=LineCompletionParameters(
            seed_distance=0.03,
            completion_radius=0.08,
            minimum_line_seed_points=5,
            minimum_line_length=1.0,
            strong_plant_margin=3,
            ransac_iterations=300,
        ),
    )

    assert not rejected[protected_id]
    assert rejected.sum() == len(coordinates) - 1


def test_planned_fence_uses_rigid_line_completion_not_seed_points_only() -> None:
    fence = np.array(
        [(x, 0.0, 1.0) for x in np.linspace(-5.0, 5.0, 101)],
        dtype=np.float64,
    )
    plant = np.array([(0.0, 2.0, 1.0)], dtype=np.float64)
    coordinates = np.vstack((fence, plant))
    rgb = np.full((len(coordinates), 3), 90, dtype=np.uint8)
    rgb[-1] = (40, 150, 35)
    fence_votes = np.zeros(len(coordinates), dtype=np.uint8)
    fence_votes[: len(fence) : 5] = 2

    rejected, report = complete_rigid_line_classes(
        coordinates,
        rgb=rgb,
        candidate_mask=np.ones(len(coordinates), dtype=bool),
        class_votes={"fence": fence_votes},
        class_plant_votes={
            "fence": np.zeros(len(coordinates), dtype=np.uint8)
        },
        parameters=LineCompletionParameters(
            seed_distance=0.03,
            completion_radius=0.08,
            minimum_line_seed_points=5,
            minimum_line_length=1.0,
            ransac_iterations=500,
            random_seed=11,
        ),
    )

    assert rejected[: len(fence)].all()
    assert not rejected[-1]
    assert report["classes"]["fence"]["completed_point_count"] == len(fence)
    assert report["completed_point_count"] == len(fence)


def test_each_rigid_class_can_use_source_verified_line_parameters() -> None:
    long_rail = np.array(
        [(x, 0.0, 1.0) for x in np.linspace(-2.0, 2.0, 41)],
        dtype=np.float64,
    )
    short_post = np.array(
        [(3.0, 0.0, z) for z in np.linspace(0.0, 0.5, 11)],
        dtype=np.float64,
    )
    coordinates = np.vstack((long_rail, short_post))
    rgb = np.full((len(coordinates), 3), 90, dtype=np.uint8)
    rail_votes = np.zeros(len(coordinates), dtype=np.uint8)
    post_votes = np.zeros(len(coordinates), dtype=np.uint8)
    rail_votes[: len(long_rail) : 5] = 2
    post_votes[len(long_rail) :: 5] = 2

    rejected, report = complete_rigid_line_classes(
        coordinates,
        rgb=rgb,
        candidate_mask=np.ones(len(coordinates), dtype=bool),
        class_votes={"rail": rail_votes, "post": post_votes},
        class_plant_votes={
            "rail": np.zeros(len(coordinates), dtype=np.uint8),
            "post": np.zeros(len(coordinates), dtype=np.uint8),
        },
        parameters=LineCompletionParameters(
            minimum_line_seed_points=5,
            minimum_line_length=1.0,
            seed_distance=0.03,
            completion_radius=0.08,
            ransac_iterations=500,
        ),
        parameters_by_class={
            "post": LineCompletionParameters(
                minimum_line_seed_points=2,
                minimum_line_length=0.3,
                seed_distance=0.03,
                completion_radius=0.08,
                ransac_iterations=500,
            )
        },
    )

    assert rejected.all()
    assert report["classes"]["rail"]["accepted_line_count"] == 1
    assert report["classes"]["post"]["accepted_line_count"] == 1


def test_source_relative_height_gate_excludes_ground_border_lines() -> None:
    ground_border = np.array(
        [(x, 0.0, 0.0) for x in np.linspace(-2.0, 2.0, 41)],
        dtype=np.float64,
    )
    raised_beam = np.array(
        [(x, 1.0, 1.5) for x in np.linspace(-2.0, 2.0, 41)],
        dtype=np.float64,
    )
    coordinates = np.vstack((ground_border, raised_beam))
    rgb = np.full((len(coordinates), 3), 90, dtype=np.uint8)
    railing_votes = np.zeros(len(coordinates), dtype=np.uint8)
    railing_votes[: len(ground_border) : 5] = 2
    railing_votes[len(ground_border) :: 5] = 2

    rejected, report = complete_railing_lines(
        coordinates,
        rgb=rgb,
        candidate_mask=np.ones(len(coordinates), dtype=bool),
        railing_votes=railing_votes,
        plant_votes=np.zeros(len(coordinates), dtype=np.uint8),
        support_height=0.0,
        parameters=LineCompletionParameters(
            minimum_seed_height_above_support=0.5,
            minimum_line_seed_points=5,
            minimum_line_length=1.0,
            seed_distance=0.03,
            completion_radius=0.08,
            ransac_iterations=500,
        ),
    )

    assert not rejected[: len(ground_border)].any()
    assert rejected[len(ground_border) :].all()
    assert report["height_eligible_seed_count"] == 9
    assert report["accepted_line_count"] == 1


def test_verified_component_grows_through_frame_but_not_ground_or_plant() -> None:
    top_beam = np.array(
        [(x, 0.0, 2.0) for x in np.linspace(-2.0, 2.0, 41)],
        dtype=np.float64,
    )
    posts = np.array(
        [
            (x, 0.0, z)
            for x in (-2.0, 2.0)
            for z in np.linspace(0.3, 2.0, 18)
        ],
        dtype=np.float64,
    )
    ground_border = np.array(
        [(x, 0.0, 0.0) for x in np.linspace(-2.0, 2.0, 41)],
        dtype=np.float64,
    )
    plant = np.array(
        [(0.0, 0.1, z) for z in np.linspace(0.3, 2.0, 18)],
        dtype=np.float64,
    )
    coordinates = np.vstack((top_beam, posts, ground_border, plant))
    frame_count = len(top_beam) + len(posts)
    ground_start = frame_count
    plant_start = ground_start + len(ground_border)
    rgb = np.full((len(coordinates), 3), 100, dtype=np.uint8)
    rgb[plant_start:] = (35, 160, 30)
    seed_mask = np.zeros(len(coordinates), dtype=bool)
    seed_mask[[0, 20, 40, 50, 68]] = True
    object_votes = np.zeros(len(coordinates), dtype=np.uint8)
    object_votes[seed_mask] = 2
    plant_votes = np.zeros(len(coordinates), dtype=np.uint8)
    plant_votes[plant_start:] = 5

    rejected, report = complete_rigid_component_classes(
        coordinates,
        rgb=rgb,
        candidate_mask=np.ones(len(coordinates), dtype=bool),
        seed_mask=seed_mask,
        class_votes={"pavilion": object_votes},
        class_plant_votes={"pavilion": plant_votes},
        support_height=0.0,
        parameters_by_class={
            "pavilion": RigidComponentCompletionParameters(
                minimum_seed_height_above_support=0.8,
                minimum_completion_height_above_support=0.2,
                voxel_size=0.25,
                bounds_margin=0.2,
            )
        },
    )

    assert rejected[:frame_count].all()
    assert not rejected[ground_start:plant_start].any()
    assert not rejected[plant_start:].any()
    assert report["classes"]["pavilion"]["completed_point_count"] == (
        frame_count
    )


def test_rejected_fence_evidence_can_complete_remaining_candidate_points() -> None:
    fence = np.array(
        [(x, 0.0, 1.0) for x in np.linspace(-5.0, 5.0, 101)],
        dtype=np.float64,
    )
    rgb = np.full((len(fence), 3), 90, dtype=np.uint8)
    fence_votes = np.zeros(len(fence), dtype=np.uint8)
    evidence_mask = np.zeros(len(fence), dtype=bool)
    evidence_mask[::5] = True
    fence_votes[evidence_mask] = 2
    candidate_mask = ~evidence_mask

    rejected, report = complete_rigid_line_classes(
        fence,
        rgb=rgb,
        candidate_mask=candidate_mask,
        seed_mask=evidence_mask,
        class_votes={"fence": fence_votes},
        class_plant_votes={
            "fence": np.zeros(len(fence), dtype=np.uint8)
        },
        parameters=LineCompletionParameters(
            seed_distance=0.03,
            completion_radius=0.08,
            minimum_line_seed_points=5,
            minimum_line_length=1.0,
            ransac_iterations=500,
            random_seed=11,
        ),
    )

    assert rejected[candidate_mask].all()
    assert not rejected[evidence_mask].any()
    assert report["classes"]["fence"]["confirmed_seed_count"] == int(
        evidence_mask.sum()
    )
    assert report["completed_point_count"] == int(candidate_mask.sum())


def test_rejected_fence_evidence_completes_a_slatted_surface_safely() -> None:
    fence = np.array(
        [
            (x, 0.0, z)
            for x in np.linspace(-5.0, 5.0, 41)
            for z in np.linspace(0.0, 3.0, 21)
        ],
        dtype=np.float64,
    )
    protected_plant = np.array(
        [(x, 0.02, 1.5) for x in np.linspace(-0.5, 0.5, 9)],
        dtype=np.float64,
    )
    detached_plant = np.array([(0.0, 2.0, 1.5)], dtype=np.float64)
    coordinates = np.vstack((fence, protected_plant, detached_plant))
    rgb = np.full((len(coordinates), 3), 90, dtype=np.uint8)
    rgb[len(fence) :] = (35, 160, 30)
    evidence_mask = np.zeros(len(coordinates), dtype=bool)
    evidence_mask[: len(fence) : 4] = True
    fence_votes = np.zeros(len(coordinates), dtype=np.uint8)
    fence_votes[evidence_mask] = 2
    plant_votes = np.zeros(len(coordinates), dtype=np.uint8)
    plant_votes[len(fence) :] = 5
    candidate_mask = ~evidence_mask

    rejected, report = complete_rigid_surface_classes(
        coordinates,
        rgb=rgb,
        candidate_mask=candidate_mask,
        seed_mask=evidence_mask,
        class_votes={"fence": fence_votes},
        class_plant_votes={"fence": plant_votes},
        parameters=RigidSurfaceCompletionParameters(
            minimum_surface_seed_points=20,
            minimum_surface_span=1.0,
            plane_distance=0.08,
        ),
    )

    assert rejected[: len(fence)][candidate_mask[: len(fence)]].all()
    assert not rejected[len(fence) :].any()
    assert report["classes"]["fence"]["accepted_surface_count"] == 1
    assert report["completed_point_count"] == int(
        candidate_mask[: len(fence)].sum()
    )
