from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image

from plant_cleanup.camera_projection import (
    CameraCalibration,
    visible_point_samples,
)
from plant_cleanup.plyio import read_cloud
from railing_removal.native_export import _native_header


class Predictor(Protocol):
    def predict(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, Mapping[int, str]]:
        """Return a semantic label image and its label mapping."""


def select_diverse_cameras(
    cameras: Sequence[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Sample aligned cameras at equal distances along the capture path."""

    if count < 1:
        raise ValueError("camera count must be positive")
    usable = [
        camera
        for camera in cameras
        if camera.get("aligned")
        and camera.get("enabled", True)
        and camera.get("center") is not None
    ]
    if len(usable) <= count:
        return usable
    centers = np.asarray(
        [camera["center"] for camera in usable],
        dtype=np.float64,
    )
    travel = np.r_[0.0, np.linalg.norm(np.diff(centers, axis=0), axis=1)]
    cumulative = np.cumsum(travel)
    if cumulative[-1] <= 1e-12:
        indices = np.linspace(0, len(usable) - 1, count)
    else:
        targets = np.linspace(0.0, cumulative[-1], count)
        indices = np.searchsorted(cumulative, targets, side="left")
    indices = np.clip(np.rint(indices).astype(np.int64), 0, len(usable) - 1)
    unique = list(dict.fromkeys(int(index) for index in indices))
    if len(unique) < count:
        for index in range(len(usable)):
            if index not in unique:
                unique.append(index)
            if len(unique) == count:
                break
        unique.sort()
    return [usable[index] for index in unique]


def _native_coordinates(
    native_cloud_path: Path,
    source_indices: np.ndarray,
) -> np.ndarray:
    count, offset = _native_header(native_cloud_path)
    if len(source_indices) and int(source_indices[-1]) >= count:
        raise ValueError("canonical source index exceeds native cloud")
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("classification", "u1"),
        ]
    )
    native = np.memmap(
        native_cloud_path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=(count,),
    )
    try:
        selected = native[source_indices]
        return np.column_stack(
            (selected["x"], selected["y"], selected["z"])
        ).astype(np.float64)
    finally:
        native._mmap.close()


def _photo_path(photo_root: Path, camera: Mapping[str, Any]) -> Path:
    recorded = Path(str(camera.get("photo") or ""))
    candidates = [
        photo_root / "png" / recorded.name,
        photo_root / recorded.name,
        photo_root / "png" / f"{camera['label']}.png",
        photo_root / f"{camera['label']}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"photo for camera {camera['label']} not found under {photo_root}"
    )


def _resize(image: Image.Image, maximum_dimension: int) -> Image.Image:
    scale = min(maximum_dimension / max(image.size), 1.0)
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def aggregate_source_photo_votes(
    canonical_cloud_path: Path,
    native_cloud_path: Path,
    inventory_path: Path,
    photo_root: Path,
    output_dir: Path,
    *,
    predictor: Predictor,
    model_id: str,
    plant_labels: tuple[str, ...],
    camera_count: int = 8,
    maximum_dimension: int = 1024,
) -> dict[str, Any]:
    """Back-project source-photo semantic labels onto canonical cloud rows."""

    canonical_cloud_path = canonical_cloud_path.resolve()
    native_cloud_path = native_cloud_path.resolve()
    inventory_path = inventory_path.resolve()
    photo_root = photo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"source-photo semantic output already exists: {output_dir}"
        )
    if maximum_dimension < 64:
        raise ValueError("maximum_dimension must be at least 64")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not inventory.get("project_opened_read_only"):
        raise ValueError("camera inventory lacks read-only provenance")
    cameras = select_diverse_cameras(
        inventory["cameras"],
        count=camera_count,
    )
    if not cameras:
        raise ValueError("camera inventory contains no usable cameras")
    cloud = read_cloud(canonical_cloud_path)
    source_indices = np.asarray(cloud["source_index"], dtype=np.uint32)
    if len(source_indices) > 1 and np.any(
        source_indices[1:] <= source_indices[:-1]
    ):
        raise ValueError("canonical source indices must be strictly increasing")
    coordinates = _native_coordinates(native_cloud_path, source_indices)
    plant_votes = np.zeros(len(cloud), dtype=np.uint8)
    background_votes = np.zeros(len(cloud), dtype=np.uint8)
    output_dir.mkdir(parents=True)
    views = []
    for camera in cameras:
        photo_path = _photo_path(photo_root, camera)
        with Image.open(photo_path) as source:
            image = _resize(source.convert("RGB"), maximum_dimension)
        labels, id2label = predictor.predict(image)
        labels = np.asarray(labels)
        if labels.shape != (image.height, image.width):
            raise ValueError(
                f"camera {camera['label']} returned mismatched labels"
            )
        plant_ids = [
            label_id
            for label_id, label in id2label.items()
            if label in plant_labels
        ]
        rows, pixels = visible_point_samples(
            coordinates,
            camera_transform=np.asarray(
                camera["transform"],
                dtype=np.float64,
            ).reshape(4, 4),
            calibration=CameraCalibration(**camera["calibration"]),
            output_width=image.width,
            output_height=image.height,
        )
        visible_labels = labels[pixels[:, 1], pixels[:, 0]]
        plant = np.isin(visible_labels, plant_ids)
        np.add.at(plant_votes, rows[plant], 1)
        np.add.at(background_votes, rows[~plant], 1)
        stem = f"camera-{camera['label']}"
        image.save(output_dir / f"{stem}-source.jpg", quality=92)
        mask = np.isin(labels, plant_ids)
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            output_dir / f"{stem}-plant-mask.png"
        )
        base = np.asarray(image, dtype=np.float32)
        color = np.empty_like(base)
        color[mask] = (40, 255, 80)
        color[~mask] = (255, 70, 40)
        overlay = (base * 0.55 + color * 0.45).astype(np.uint8)
        Image.fromarray(overlay).save(
            output_dir / f"{stem}-overlay.jpg",
            quality=92,
        )
        views.append(
            {
                "camera": str(camera["label"]),
                "photo": str(photo_path),
                "image_size": [image.width, image.height],
                "visible_point_count": int(len(rows)),
                "plant_point_count": int(plant.sum()),
                "background_point_count": int((~plant).sum()),
            }
        )
    np.save(output_dir / "plant-votes.npy", plant_votes)
    np.save(output_dir / "background-votes.npy", background_votes)
    report = {
        "schema_version": 1,
        "method": "calibrated-source-photo-semantic-backprojection-v1",
        "model": model_id,
        "plant_labels": list(plant_labels),
        "canonical_cloud": str(canonical_cloud_path),
        "native_cloud": str(native_cloud_path),
        "camera_inventory": str(inventory_path),
        "project_opened_read_only": True,
        "point_count": int(len(cloud)),
        "camera_count": len(cameras),
        "maximum_dimension": maximum_dimension,
        "plant_seed_count": int(
            np.count_nonzero(plant_votes > background_votes)
        ),
        "background_seed_count": int(
            np.count_nonzero(background_votes > plant_votes)
        ),
        "unseen_point_count": int(
            np.count_nonzero((plant_votes + background_votes) == 0)
        ),
        "views": views,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
