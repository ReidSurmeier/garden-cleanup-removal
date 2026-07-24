from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from plant_cleanup.plyio import read_cloud


BACKGROUND = np.array([18, 18, 22], dtype=np.uint8)
DEFAULT_MAX_RENDER_POINTS = 1_200_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_cloud_for_render(
    cloud: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, float]:
    """Bound proof-only raster work without changing full cloud exports."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(cloud) <= max_points:
        return cloud, 1.0
    if max_points == 1:
        return cloud[[(len(cloud) - 1) // 2]], float(len(cloud))
    indices = np.linspace(
        0,
        len(cloud) - 1,
        num=max_points,
        dtype=np.int64,
    )
    return cloud[indices], (len(cloud) - 1) / (max_points - 1)


def _canonical_frame(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a deterministic horizontal frame while preserving Metashape's Z-up axis."""
    if len(coordinates) == 0:
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    center = np.median(coordinates, axis=0)
    horizontal = coordinates[:, :2] - center[:2]
    covariance = horizontal.T @ horizontal / max(len(horizontal), 1)
    values, vectors = np.linalg.eigh(covariance)
    major_xy = vectors[:, np.argmax(values)]
    largest_component = int(np.argmax(np.abs(major_xy)))
    if major_xy[largest_component] < 0:
        major_xy = -major_xy
    minor_xy = np.array([-major_xy[1], major_xy[0]])
    axes = np.array(
        [
            [major_xy[0], minor_xy[0], 0.0],
            [major_xy[1], minor_xy[1], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return center, axes


def _rasterize(
    projected: np.ndarray,
    rgb: np.ndarray,
    source_ids: np.ndarray,
    *,
    size: int,
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    if len(projected) == 0:
        return (
            np.broadcast_to(BACKGROUND, (size, size, 3)).copy(),
            np.full((size, size), np.inf, dtype=np.float32),
            np.full((size, size), -1, dtype=np.int64),
            {
                "horizontal_center": 0.0,
                "vertical_center": 0.0,
                "scale": 0.0,
                "size": size,
                "scene_extent": 0.0,
            },
        )
    horizontal = projected[:, 0]
    vertical = projected[:, 1]
    depth = projected[:, 2]

    horizontal_center = (float(horizontal.min()) + float(horizontal.max())) / 2.0
    vertical_center = (float(vertical.min()) + float(vertical.max())) / 2.0
    extent = max(float(np.ptp(horizontal)), float(np.ptp(vertical)), 1e-9)
    scene_extent = max(
        float(np.ptp(horizontal)),
        float(np.ptp(vertical)),
        float(np.ptp(depth)),
        1e-9,
    )
    scale = (size - 1) * 0.94 / extent
    x = np.rint((horizontal - horizontal_center) * scale + (size - 1) / 2).astype(
        np.int32
    )
    y = np.rint((vertical_center - vertical) * scale + (size - 1) / 2).astype(
        np.int32
    )

    image = np.broadcast_to(BACKGROUND, (size, size, 3)).copy()
    depth_buffer = np.full((size, size), -np.inf, dtype=np.float32)
    id_buffer = np.full((size, size), -1, dtype=np.int64)

    offsets = [
        (dx, dy)
        for dy in range(-point_radius, point_radius + 1)
        for dx in range(-point_radius, point_radius + 1)
        if dx * dx + dy * dy <= point_radius * point_radius
    ]
    for dx, dy in offsets:
        target_x = x + dx
        target_y = y + dy
        valid = (
            (target_x >= 0)
            & (target_x < size)
            & (target_y >= 0)
            & (target_y < size)
        )
        indices = np.flatnonzero(valid)
        pixels = target_y[indices].astype(np.int64) * size + target_x[indices]
        order = np.lexsort((source_ids[indices], depth[indices], pixels))
        ordered_pixels = pixels[order]
        last = np.r_[ordered_pixels[1:] != ordered_pixels[:-1], True]
        winners = indices[order[last]]
        winner_y = target_y[winners]
        winner_x = target_x[winners]
        nearer = depth[winners] > depth_buffer[winner_y, winner_x]
        winners = winners[nearer]
        winner_y = target_y[winners]
        winner_x = target_x[winners]
        depth_buffer[winner_y, winner_x] = depth[winners]
        id_buffer[winner_y, winner_x] = source_ids[winners]
        image[winner_y, winner_x] = rgb[winners]

    depth_output = np.full_like(depth_buffer, np.inf)
    visible = id_buffer >= 0
    depth_output[visible] = depth_buffer[visible]
    return image, depth_output, id_buffer, {
        "horizontal_center": horizontal_center,
        "vertical_center": vertical_center,
        "scale": scale,
        "size": size,
        "scene_extent": scene_extent,
    }


def render_cloud_views(
    cloud_path: Path,
    output_dir: Path,
    *,
    size: int = 1200,
    point_radius: int = 1,
    yaw_degrees: tuple[int, ...] | None = None,
    max_points: int = DEFAULT_MAX_RENDER_POINTS,
) -> dict[str, Any]:
    """Render repeatable RGB/depth/source-ID proof views of the PLY contract."""
    source_cloud = read_cloud(cloud_path)
    cloud, sampling_step = _sample_cloud_for_render(
        source_cloud,
        max_points=max_points,
    )
    coordinates = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(
        np.float64
    )
    colors = np.column_stack((cloud["red"], cloud["green"], cloud["blue"]))
    source_ids = np.asarray(cloud["source_index"], dtype=np.int64)
    center, axes = _canonical_frame(coordinates)
    canonical = (coordinates - center) @ axes
    if yaw_degrees is None:
        projections = [
            ("front", canonical[:, (0, 2, 1)]),
            ("side", canonical[:, (1, 2, 0)]),
            ("top", canonical[:, (0, 1, 2)]),
        ]
    else:
        projections = []
        for degrees in yaw_degrees:
            radians = math.radians(degrees)
            cosine = math.cos(radians)
            sine = math.sin(radians)
            horizontal = cosine * canonical[:, 0] + sine * canonical[:, 1]
            depth = -sine * canonical[:, 0] + cosine * canonical[:, 1]
            projections.append(
                (
                    f"orbit-{degrees % 360:03d}",
                    np.column_stack((horizontal, canonical[:, 2], depth)),
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    views: list[dict[str, Any]] = []
    for name, projected in projections:
        image, depths, ids, projection = _rasterize(
            projected,
            colors,
            source_ids,
            size=size,
            point_radius=point_radius,
        )
        rgb_path = output_dir / f"{name}-rgb.png"
        depth_path = output_dir / f"{name}-depth.npy"
        ids_path = output_dir / f"{name}-source-ids.npy"
        Image.fromarray(image, mode="RGB").save(rgb_path)
        np.save(depth_path, depths)
        np.save(ids_path, ids)
        views.append(
            {
                "view": name,
                "rgb": str(rgb_path),
                "depth": str(depth_path),
                "source_ids": str(ids_path),
                "projection": projection,
                "visible_pixels": int(np.count_nonzero(ids >= 0)),
                "rgb_sha256": _sha256(rgb_path),
            }
        )
    report = {
        "cloud": str(cloud_path),
        "point_count": int(len(source_cloud)),
        "rendered_point_count": int(len(cloud)),
        "render_sampling_step": sampling_step,
        "quality_state": "usable" if len(source_cloud) else "unusable",
        "quality_reason": None if len(source_cloud) else "empty_cloud",
        "frame": {
            "center": center.tolist(),
            "axes": axes.tolist(),
            "z_up": True,
        },
        "views": views,
    }
    if yaw_degrees is not None:
        report["yaw_degrees"] = list(yaw_degrees)
    return report
