from __future__ import annotations

import numpy as np

from plant_cleanup.dense_semantic import (
    DensePropagationParameters,
    propagate_dense_semantic_evidence,
)


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
