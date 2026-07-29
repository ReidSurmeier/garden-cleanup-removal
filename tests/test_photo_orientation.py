from __future__ import annotations

import numpy as np
import cv2

from railing_removal.photo_orientation import (
    CameraProjection,
    LineSegment,
    detect_vertical_line_segments,
    estimate_vertical_from_segments,
)


def test_photo_line_constraints_recover_vertical_despite_diagonal_outliers() -> None:
    camera = CameraProjection(
        right=np.array((1.0, 0.0, 0.0)),
        down=np.array((0.0, 1.0, 0.0)),
        forward=np.array((0.0, 0.0, 1.0)),
        focal_length=1000.0,
        principal_x=500.0,
        principal_y=500.0,
    )
    segments = [
        LineSegment(x1=x, y1=100.0, x2=x, y2=900.0, weight=800.0)
        for x in (180.0, 300.0, 450.0, 620.0, 790.0)
    ]
    segments.extend(
        [
            LineSegment(100.0, 100.0, 900.0, 500.0, 894.0),
            LineSegment(150.0, 800.0, 850.0, 200.0, 922.0),
        ]
    )

    result = estimate_vertical_from_segments(
        [(camera, segment) for segment in segments],
        reference_up=np.array((0.0, -1.0, 0.0)),
        maximum_constraint_error_degrees=3.0,
        minimum_inlier_segments=4,
    )

    assert result["status"] == "usable"
    assert result["inlier_segment_count"] == 5
    np.testing.assert_allclose(
        result["up"],
        (0.0, -1.0, 0.0),
        atol=1e-6,
    )


def test_photo_orientation_fails_closed_without_enough_consistent_lines() -> None:
    camera = CameraProjection(
        right=np.array((1.0, 0.0, 0.0)),
        down=np.array((0.0, 1.0, 0.0)),
        forward=np.array((0.0, 0.0, 1.0)),
        focal_length=1000.0,
        principal_x=500.0,
        principal_y=500.0,
    )

    result = estimate_vertical_from_segments(
        [
            (camera, LineSegment(100.0, 100.0, 100.0, 900.0, 800.0)),
            (camera, LineSegment(900.0, 100.0, 900.0, 900.0, 800.0)),
        ],
        reference_up=np.array((0.0, -1.0, 0.0)),
        minimum_inlier_segments=4,
    )

    assert result["status"] == "insufficient_evidence"


def test_detector_keeps_long_image_verticals_and_rejects_diagonals() -> None:
    image = np.zeros((1000, 1000), dtype=np.uint8)
    for x in (150, 300, 450, 600, 750):
        cv2.line(image, (x, 100), (x, 900), 255, 8)
    cv2.line(image, (50, 100), (950, 600), 255, 8)
    cv2.line(image, (50, 900), (950, 300), 255, 8)

    segments = detect_vertical_line_segments(
        image,
        maximum_image_vertical_deviation_degrees=15.0,
        minimum_length_fraction=0.25,
    )

    assert len(segments) >= 5
    assert all(
        np.degrees(
            np.arctan2(
                abs(segment.x2 - segment.x1),
                abs(segment.y2 - segment.y1),
            )
        )
        <= 15.0
        for segment in segments
    )
