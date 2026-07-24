from __future__ import annotations

import numpy as np

from plant_cleanup.camera_projection import (
    CameraCalibration,
    project_chunk_points,
    visible_point_samples,
)


def test_vectorized_camera_projection_matches_metashape_reference() -> None:
    calibration = CameraCalibration(
        width=2160,
        height=3840,
        f=2920.5177526197276,
        cx=-14.939502928756955,
        cy=5.846871277028682,
        b1=0.0,
        b2=0.0,
        k1=0.22248522655950406,
        k2=-0.7020140403115382,
        k3=0.7035614537819277,
        k4=0.0,
        p1=0.0027130191225739793,
        p2=0.0009338966702340011,
    )
    transform = np.array(
        [
            [0.9919369399353384, -0.12581351890761822,
             -0.015233701185198102, -5.467075710196299],
            [-0.11652390505250511, -0.9526871561422409,
             0.28073005195903755, 0.20816631457121698],
            [-0.0498325871597321, -0.2766914183376339,
             -0.9596658649109471, 0.14386334355428676],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    point = np.array(
        [[-7.072013296153882, 2.5068358165101627, -4.189592948461738]],
        dtype=np.float64,
    )

    pixels, depth, visible = project_chunk_points(
        point,
        camera_transform=transform,
        calibration=calibration,
    )

    np.testing.assert_allclose(
        pixels[0],
        [54.62130173380626, 1440.7514661081582],
        atol=1e-9,
    )
    np.testing.assert_allclose(depth[0], 4.828424828852691, atol=1e-12)
    assert visible.tolist() == [True]


def test_projection_marks_points_outside_the_photo_or_behind_camera() -> None:
    calibration = CameraCalibration(
        width=100,
        height=80,
        f=50.0,
    )
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [10.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    pixels, depth, visible = project_chunk_points(
        points,
        camera_transform=np.eye(4),
        calibration=calibration,
    )

    np.testing.assert_allclose(pixels[0], [50.0, 40.0])
    np.testing.assert_allclose(depth, [2.0, 1.0, -1.0])
    assert visible.tolist() == [True, False, False]


def test_visible_samples_keep_the_nearest_point_per_segmentation_pixel() -> None:
    calibration = CameraCalibration(
        width=100,
        height=80,
        f=50.0,
    )
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.8, 0.0, 2.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    rows, pixels = visible_point_samples(
        points,
        camera_transform=np.eye(4),
        calibration=calibration,
        output_width=50,
        output_height=40,
    )

    assert rows.tolist() == [0, 2]
    assert pixels.tolist() == [[25, 20], [35, 20]]
