from __future__ import annotations

import numpy as np

from plant_cleanup.source_photo_semantic import (
    fuse_source_photo_votes,
    select_diverse_cameras,
)


def test_camera_selection_uses_travel_distance_instead_of_frame_count() -> None:
    cameras = [
        {"label": "1", "center": [0.0, 0.0, 0.0], "aligned": True},
        {"label": "2", "center": [0.1, 0.0, 0.0], "aligned": True},
        {"label": "3", "center": [0.2, 0.0, 0.0], "aligned": True},
        {"label": "4", "center": [5.0, 0.0, 0.0], "aligned": True},
        {"label": "5", "center": [10.0, 0.0, 0.0], "aligned": True},
    ]

    selected = select_diverse_cameras(cameras, count=3)

    assert [camera["label"] for camera in selected] == ["1", "4", "5"]


def test_source_photo_votes_override_fallback_only_where_observed() -> None:
    plant, background, source_seen = fuse_source_photo_votes(
        source_plant=np.array([1, 0, 1], dtype=np.uint8),
        source_background=np.array([0, 0, 1], dtype=np.uint8),
        fallback_plant=np.array([0, 3, 5], dtype=np.uint8),
        fallback_background=np.array([5, 0, 0], dtype=np.uint8),
    )

    assert plant.tolist() == [1, 3, 1]
    assert background.tolist() == [0, 0, 1]
    assert source_seen.tolist() == [True, False, True]
