from __future__ import annotations

from plant_cleanup.source_photo_semantic import select_diverse_cameras


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
