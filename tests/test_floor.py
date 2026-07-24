from __future__ import annotations

import numpy as np

from railing_removal.floor import FloorRemovalParameters, remove_uncertain_floor


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
