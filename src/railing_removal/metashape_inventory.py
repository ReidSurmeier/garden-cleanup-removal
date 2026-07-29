from __future__ import annotations

from pathlib import Path
from typing import Any

from railing_removal.metashape_reader import (
    _camera_coordinate_frame,
    _reframe,
)


def _vector(value: object | None) -> list[float] | None:
    return [float(item) for item in value] if value is not None else None


def _float(value: object | None) -> float | None:
    return float(value) if value is not None else None


def _camera_record(camera: Any, coordinate_frame: dict[str, Any]) -> dict[str, Any]:
    calibration = (
        camera.sensor.calibration
        if getattr(camera, "sensor", None) is not None
        else None
    )
    transform = getattr(camera, "transform", None)
    source_frame_center = None
    source_frame_up = None
    source_frame_right = None
    source_frame_down = None
    source_frame_forward = None
    if transform is not None:
        matrix = list(transform)
        source_frame_center = list(
            _reframe(camera.center, coordinate_frame)
        )
        source_frame_up = list(
            _reframe(
                (-matrix[1], -matrix[5], -matrix[9]),
                coordinate_frame,
                normalize=True,
            )
        )
        source_frame_right = list(
            _reframe(
                (matrix[0], matrix[4], matrix[8]),
                coordinate_frame,
                normalize=True,
            )
        )
        source_frame_down = list(
            _reframe(
                (matrix[1], matrix[5], matrix[9]),
                coordinate_frame,
                normalize=True,
            )
        )
        source_frame_forward = list(
            _reframe(
                (matrix[2], matrix[6], matrix[10]),
                coordinate_frame,
                normalize=True,
            )
        )
    reference = getattr(camera, "reference", None)
    return {
        "label": str(camera.label),
        "enabled": bool(camera.enabled),
        "aligned": transform is not None,
        "photo": (
            str(camera.photo.path)
            if getattr(camera, "photo", None) is not None
            else None
        ),
        "center": _vector(getattr(camera, "center", None)),
        "transform": _vector(transform),
        "source_frame_center": source_frame_center,
        "source_frame_up": source_frame_up,
        "source_frame_right": source_frame_right,
        "source_frame_down": source_frame_down,
        "source_frame_forward": source_frame_forward,
        "reference": (
            {
                "enabled": bool(getattr(reference, "enabled", False)),
                "location": _vector(getattr(reference, "location", None)),
                "location_accuracy": _vector(
                    getattr(reference, "location_accuracy", None)
                ),
            }
            if reference is not None
            else None
        ),
        "calibration": (
            {
                "width": int(calibration.width),
                "height": int(calibration.height),
                "f": float(calibration.f),
                "cx": float(calibration.cx),
                "cy": float(calibration.cy),
                "b1": float(calibration.b1),
                "b2": float(calibration.b2),
                "k1": float(calibration.k1),
                "k2": float(calibration.k2),
                "k3": float(calibration.k3),
                "k4": float(calibration.k4),
                "p1": float(calibration.p1),
                "p2": float(calibration.p2),
            }
            if calibration is not None
            else None
        ),
    }


def _scale_bar_record(scale_bar: Any) -> dict[str, Any]:
    reference = getattr(scale_bar, "reference", None)
    return {
        "label": str(scale_bar.label),
        "enabled": bool(getattr(reference, "enabled", False)),
        "distance": _float(getattr(reference, "distance", None)),
        "accuracy": _float(getattr(reference, "accuracy", None)),
    }


def build_camera_inventory(
    chunk: Any,
    *,
    project: Path,
    metashape_version: str,
    project_opened_read_only: bool,
) -> dict[str, Any]:
    if not project_opened_read_only:
        raise ValueError("camera inventory requires a read-only project")
    coordinate_frame = _camera_coordinate_frame(chunk)
    cameras = [
        _camera_record(camera, coordinate_frame)
        for camera in chunk.cameras
    ]
    scale_bars = [
        _scale_bar_record(scale_bar)
        for scale_bar in getattr(chunk, "scalebars", [])
    ]
    transform = getattr(chunk, "transform", None)
    crs = getattr(chunk, "crs", None)
    return {
        "schema_version": 3,
        "metashape_version": str(metashape_version),
        "project": str(project),
        "project_opened_read_only": True,
        "chunk": str(chunk.label),
        "point_count": (
            int(chunk.point_cloud.point_count)
            if getattr(chunk, "point_cloud", None) is not None
            else 0
        ),
        "camera_count": len(cameras),
        "aligned_camera_count": sum(
            camera["aligned"] for camera in cameras
        ),
        "depth_map_count": len(chunk.depth_maps.keys()),
        "coordinate_frame": coordinate_frame,
        "chunk_transform": {
            "scale": _float(getattr(transform, "scale", None)),
            "matrix": _vector(getattr(transform, "matrix", None)),
        },
        "coordinate_reference": (
            {
                "name": str(getattr(crs, "name", str(crs))),
                "value": str(crs),
            }
            if crs is not None
            else None
        ),
        "scale_bars": scale_bars,
        "reference_summary": {
            "enabled_camera_locations": sum(
                bool((camera.get("reference") or {}).get("enabled"))
                and (camera.get("reference") or {}).get("location")
                is not None
                for camera in cameras
            ),
            "enabled_scale_bars": sum(
                scale_bar["enabled"]
                and scale_bar["distance"] is not None
                for scale_bar in scale_bars
            ),
        },
        "cameras": cameras,
    }
