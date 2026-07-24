from __future__ import annotations

import numpy as np

from railing_removal import LineCompletionParameters, complete_railing_lines


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
