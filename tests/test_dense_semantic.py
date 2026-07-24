from __future__ import annotations

import numpy as np

from plant_cleanup.dense_semantic import (
    DensePropagationParameters,
    build_scene_plant_protection,
    propagate_dense_semantic_evidence,
    select_scene_dense_rejects,
    select_dense_plant_labels,
)


def test_turf_scene_treats_closed_set_grass_as_ground_not_generic_plant(
) -> None:
    configured = ("plant", "tree", "grass", "flower")

    assert select_dense_plant_labels(
        configured,
        scene_class_ids={"turf_ground"},
    ) == ("plant", "tree", "flower")
    assert select_dense_plant_labels(
        configured,
        scene_class_ids={"pavement_ground"},
    ) == configured


def test_dense_semantics_remove_ground_and_preserve_raised_plant_structure(
) -> None:
    ground = np.array(
        [(float(x), 0.0, 0.0) for x in range(20)],
        dtype=np.float32,
    )
    plant = np.array(
        [(10.0, 0.0, float(z) / 10.0) for z in range(1, 71)],
        dtype=np.float32,
    )
    coordinates = np.vstack((ground, plant))
    normals = np.tile((0.0, 0.0, 1.0), (len(coordinates), 1))
    normals[len(ground) :] = (1.0, 0.0, 0.0)
    plant_votes = np.zeros(len(coordinates), dtype=np.uint8)
    background_votes = np.zeros(len(coordinates), dtype=np.uint8)
    plant_votes[len(ground) + 4 :: 5] = 2
    background_votes[: len(ground) : 2] = 2
    # A model-mislabeled horizontal ground point must not become a plant seed.
    plant_votes[5] = 3

    strict, conservative, report = propagate_dense_semantic_evidence(
        coordinates,
        normals=normals,
        candidate_mask=np.ones(len(coordinates), dtype=bool),
        plant_votes=plant_votes,
        background_votes=background_votes,
        support_plane_coefficients=(0.0, 0.0, 0.0),
        vertical_span=8.0,
        parameters=DensePropagationParameters(
            conservative_background_factor=1.0,
            strict_background_factor=2.0,
            ground_height_fraction=0.1,
            minimum_seed_points=2,
        ),
    )

    assert not strict[: len(ground)].any()
    assert strict[len(ground) :].all()
    assert np.all(strict <= conservative)
    assert report["ground_reclassified_seed_count"] == 1


def test_dense_semantics_preserve_explicit_scene_plant_protection() -> None:
    ground = np.array(
        [(float(x), 0.0, 0.0) for x in range(20)],
        dtype=np.float32,
    )
    plant = np.array(
        [(10.0, 0.0, float(z)) for z in range(1, 6)],
        dtype=np.float32,
    )
    coordinates = np.vstack((ground, plant))
    normals = np.tile((0.0, 0.0, 1.0), (len(coordinates), 1))
    normals[len(ground) :] = (1.0, 0.0, 0.0)
    plant_votes = np.zeros(len(coordinates), dtype=np.uint8)
    background_votes = np.zeros(len(coordinates), dtype=np.uint8)
    background_votes[: len(ground)] = 3
    plant_votes[len(ground) :] = 3
    protected_groundcover = np.zeros(len(coordinates), dtype=bool)
    protected_groundcover[10] = True

    strict, conservative, report = propagate_dense_semantic_evidence(
        coordinates,
        normals=normals,
        candidate_mask=np.ones(len(coordinates), dtype=bool),
        plant_votes=plant_votes,
        background_votes=background_votes,
        protection_mask=protected_groundcover,
        support_plane_coefficients=(0.0, 0.0, 0.0),
        vertical_span=5.0,
        parameters=DensePropagationParameters(minimum_seed_points=2),
    )

    assert strict[10]
    assert conservative[10]
    assert not strict[:10].any()
    assert not strict[11:len(ground)].any()
    assert report["explicit_protected_point_count"] == 1


def test_scene_plant_protection_requires_a_strong_class_margin() -> None:
    protection = build_scene_plant_protection(
        class_votes={
            "turf_ground": np.array([0, 2, 1, 0], dtype=np.uint8),
            "pavement_ground": np.array([0, 0, 3, 0], dtype=np.uint8),
        },
        class_plant_votes={
            "turf_ground": np.array([3, 3, 0, 0], dtype=np.uint8),
            "pavement_ground": np.array([0, 0, 6, 1], dtype=np.uint8),
        },
        minimum_margin=2,
    )

    assert protection.tolist() == [True, False, True, False]


def test_ground_scene_persists_dense_background_rejections() -> None:
    candidate = np.array([True, True, True, False], dtype=bool)
    strict_keep = np.array([True, False, True, False], dtype=bool)

    assert select_scene_dense_rejects(
        candidate,
        strict_keep,
        has_ground_surface=True,
    ).tolist() == [False, True, False, False]
    assert not select_scene_dense_rejects(
        candidate,
        strict_keep,
        has_ground_surface=False,
    ).any()
