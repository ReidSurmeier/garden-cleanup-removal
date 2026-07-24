from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    f: float
    cx: float = 0.0
    cy: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


def project_chunk_points(
    points: np.ndarray,
    *,
    camera_transform: np.ndarray,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project chunk-space points using Metashape's calibrated camera model."""

    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(camera_transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (point_count, 3)")
    if transform.shape != (4, 4):
        raise ValueError("camera_transform must have shape (4, 4)")
    camera_points = (
        points - transform[:3, 3]
    ) @ transform[:3, :3]
    depth = camera_points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = camera_points[:, 0] / depth
        y = camera_points[:, 1] / depth
        radius_squared = x * x + y * y
        radial = (
            1.0
            + calibration.k1 * radius_squared
            + calibration.k2 * radius_squared**2
            + calibration.k3 * radius_squared**3
            + calibration.k4 * radius_squared**4
        )
        distorted_x = (
            x * radial
            + calibration.p1 * (radius_squared + 2.0 * x * x)
            + 2.0 * calibration.p2 * x * y
        )
        distorted_y = (
            y * radial
            + calibration.p2 * (radius_squared + 2.0 * y * y)
            + 2.0 * calibration.p1 * x * y
        )
        pixel_x = (
            calibration.width / 2.0
            + calibration.cx
            + (calibration.f + calibration.b1) * distorted_x
            + calibration.b2 * distorted_y
        )
        pixel_y = (
            calibration.height / 2.0
            + calibration.cy
            + calibration.f * distorted_y
        )
    pixels = np.column_stack((pixel_x, pixel_y))
    visible = (
        (depth > 0.0)
        & np.isfinite(pixels).all(axis=1)
        & (pixel_x >= 0.0)
        & (pixel_x < calibration.width)
        & (pixel_y >= 0.0)
        & (pixel_y < calibration.height)
    )
    return pixels, depth, visible


def visible_point_samples(
    points: np.ndarray,
    *,
    camera_transform: np.ndarray,
    calibration: CameraCalibration,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return front-most point rows and pixels at segmentation resolution."""

    if output_width < 1 or output_height < 1:
        raise ValueError("output dimensions must be positive")
    pixels, depth, in_photo = project_chunk_points(
        points,
        camera_transform=camera_transform,
        calibration=calibration,
    )
    rows = np.flatnonzero(in_photo)
    if not len(rows):
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.int32),
        )
    pixel_x = np.floor(
        pixels[rows, 0] * output_width / calibration.width
    ).astype(np.int32)
    pixel_y = np.floor(
        pixels[rows, 1] * output_height / calibration.height
    ).astype(np.int32)
    pixel_ids = pixel_y.astype(np.int64) * output_width + pixel_x
    depth_buffer = np.full(
        output_width * output_height,
        np.inf,
        dtype=np.float64,
    )
    np.minimum.at(depth_buffer, pixel_ids, depth[rows])
    frontmost = depth[rows] <= depth_buffer[pixel_ids]
    rows = rows[frontmost]
    sampled_pixels = np.column_stack(
        (pixel_x[frontmost], pixel_y[frontmost])
    )
    return rows, sampled_pixels
