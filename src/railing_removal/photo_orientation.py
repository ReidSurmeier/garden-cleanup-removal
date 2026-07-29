from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(length) or length < 1e-12:
        raise ValueError("expected a finite nonzero 3D vector")
    return vector / length


@dataclass(frozen=True)
class CameraProjection:
    right: np.ndarray
    down: np.ndarray
    forward: np.ndarray
    focal_length: float
    principal_x: float
    principal_y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "right", _unit(self.right))
        object.__setattr__(self, "down", _unit(self.down))
        object.__setattr__(self, "forward", _unit(self.forward))
        if not np.isfinite(self.focal_length) or self.focal_length <= 0:
            raise ValueError("camera focal length must be positive")
        if not np.isfinite(self.principal_x) or not np.isfinite(
            self.principal_y
        ):
            raise ValueError("camera principal point must be finite")

    def ray(self, x: float, y: float) -> np.ndarray:
        return _unit(
            self.right * ((x - self.principal_x) / self.focal_length)
            + self.down * ((y - self.principal_y) / self.focal_length)
            + self.forward
        )


@dataclass(frozen=True)
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    weight: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2, self.weight)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("line segment values must be finite")
        if self.weight <= 0:
            raise ValueError("line segment weight must be positive")
        if np.hypot(self.x2 - self.x1, self.y2 - self.y1) < 1e-6:
            raise ValueError("line segment endpoints must differ")


def detect_vertical_line_segments(
    image: np.ndarray,
    *,
    maximum_image_vertical_deviation_degrees: float = 35.0,
    minimum_length_fraction: float = 0.08,
    maximum_dimension: int = 1600,
    maximum_segments: int = 80,
) -> list[LineSegment]:
    """Detect long, approximately image-vertical line constraints."""

    import cv2

    image = np.asarray(image)
    if image.ndim not in {2, 3}:
        raise ValueError("image must be grayscale or color")
    if not 0 < maximum_image_vertical_deviation_degrees < 90:
        raise ValueError("vertical deviation must be between 0 and 90")
    if not 0 < minimum_length_fraction < 1:
        raise ValueError("minimum length fraction must be in (0, 1)")
    if maximum_dimension < 64 or maximum_segments < 1:
        raise ValueError("detector limits are invalid")

    height, width = image.shape[:2]
    scale = min(1.0, maximum_dimension / max(height, width))
    if scale < 1.0:
        working = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = image
    if working.ndim == 3:
        gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
    else:
        gray = working.astype(np.uint8, copy=False)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    diagonal = float(np.hypot(*gray.shape))
    minimum_length = minimum_length_fraction * diagonal
    detected = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 360.0,
        threshold=max(20, round(minimum_length * 0.15)),
        minLineLength=minimum_length,
        maxLineGap=max(4, round(diagonal * 0.02)),
    )
    if detected is None:
        return []

    inverse_scale = 1.0 / scale
    result: list[LineSegment] = []
    for raw in np.asarray(detected).reshape((-1, 4)):
        x1, y1, x2, y2 = (float(value) for value in raw)
        dx, dy = x2 - x1, y2 - y1
        deviation = float(
            np.degrees(np.arctan2(abs(dx), abs(dy)))
        )
        if deviation > maximum_image_vertical_deviation_degrees:
            continue
        length = float(np.hypot(dx, dy)) * inverse_scale
        result.append(
            LineSegment(
                x1=x1 * inverse_scale,
                y1=y1 * inverse_scale,
                x2=x2 * inverse_scale,
                y2=y2 * inverse_scale,
                weight=length,
            )
        )
    result.sort(key=lambda segment: -segment.weight)
    return result[:maximum_segments]


def _constraint_normal(
    camera: CameraProjection,
    segment: LineSegment,
) -> np.ndarray:
    return _unit(
        np.cross(
            camera.ray(segment.x1, segment.y1),
            camera.ray(segment.x2, segment.y2),
        )
    )


def _constraint_errors_degrees(
    normals: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    return np.degrees(
        np.arcsin(
            np.clip(
                np.abs(normals @ _unit(candidate)),
                0.0,
                1.0,
            )
        )
    )


def estimate_vertical_from_segments(
    observations: Iterable[tuple[CameraProjection, LineSegment]],
    *,
    reference_up: np.ndarray,
    maximum_constraint_error_degrees: float = 5.0,
    minimum_inlier_segments: int = 6,
) -> dict[str, Any]:
    """Recover an axial 3D vertical from multi-view image line planes."""

    items = list(observations)
    if not 0 < maximum_constraint_error_degrees < 45:
        raise ValueError("constraint error must be between 0 and 45 degrees")
    if minimum_inlier_segments < 2:
        raise ValueError("at least two inlier segments are required")
    if len(items) < minimum_inlier_segments:
        return {
            "schema_version": 1,
            "status": "insufficient_evidence",
            "segment_count": len(items),
            "inlier_segment_count": 0,
        }

    normals = np.vstack(
        [_constraint_normal(camera, segment) for camera, segment in items]
    )
    weights = np.asarray(
        [segment.weight for _, segment in items],
        dtype=np.float64,
    )
    best_mask: np.ndarray | None = None
    best_key: tuple[int, float, float] | None = None
    for left, right in combinations(normals, 2):
        axis = np.cross(left, right)
        if float(np.linalg.norm(axis)) < 1e-8:
            continue
        errors = _constraint_errors_degrees(normals, axis)
        mask = errors <= maximum_constraint_error_degrees
        key = (
            int(np.count_nonzero(mask)),
            float(np.sum(weights[mask])),
            -float(np.median(errors[mask])) if np.any(mask) else -90.0,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_mask = mask
    if best_mask is None or np.count_nonzero(best_mask) < minimum_inlier_segments:
        return {
            "schema_version": 1,
            "status": "insufficient_evidence",
            "segment_count": len(items),
            "inlier_segment_count": (
                int(np.count_nonzero(best_mask))
                if best_mask is not None
                else 0
            ),
        }

    inlier_normals = normals[best_mask]
    inlier_weights = weights[best_mask]
    scatter = np.einsum(
        "i,ij,ik->jk",
        inlier_weights,
        inlier_normals,
        inlier_normals,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    up = _unit(eigenvectors[:, 0])
    reference = _unit(reference_up)
    if float(up @ reference) < 0:
        up = -up
    errors = _constraint_errors_degrees(inlier_normals, up)
    return {
        "schema_version": 1,
        "status": "usable",
        "up": up.tolist(),
        "segment_count": len(items),
        "inlier_segment_count": int(np.count_nonzero(best_mask)),
        "median_constraint_error_degrees": float(np.median(errors)),
        "maximum_constraint_error_degrees": float(np.max(errors)),
        "eigenvalues": eigenvalues.tolist(),
    }
