from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from plant_cleanup.plyio import read_cloud


class Predictor(Protocol):
    def predict(self, image: Image.Image) -> tuple[np.ndarray, Mapping[int, str]]:
        """Return a semantic label image and its integer label mapping."""


class HuggingFaceOneFormerPredictor:
    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import (
            OneFormerForUniversalSegmentation,
            OneFormerProcessor,
        )

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = OneFormerProcessor.from_pretrained(model_id)
        self._model = (
            OneFormerForUniversalSegmentation.from_pretrained(model_id)
            .eval()
            .to(self._device)
        )

    def predict(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, Mapping[int, str]]:
        inputs = self._processor(
            images=image,
            task_inputs=["semantic"],
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        labels = self._processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],
        )[0]
        return labels.cpu().numpy(), self._model.config.id2label


def aggregate_dense_semantic_votes(
    cloud_path: Path,
    render_dir: Path,
    output_dir: Path,
    *,
    predictor: Predictor,
    model_id: str,
    plant_labels: tuple[str, ...],
    background_labels: tuple[str, ...],
) -> dict[str, Any]:
    """Project closed-set semantic labels back to stable cloud source IDs."""

    cloud = read_cloud(cloud_path.resolve())
    source_indices = np.asarray(cloud["source_index"])
    if len(source_indices) and np.any(source_indices[1:] < source_indices[:-1]):
        raise ValueError("cloud source_index values must be sorted")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    plant_votes = np.zeros(len(cloud), dtype=np.uint8)
    background_votes = np.zeros(len(cloud), dtype=np.uint8)
    view_reports: list[dict[str, Any]] = []

    for image_path in sorted(render_dir.resolve().glob("orbit-*-rgb.png")):
        stem = image_path.name.removesuffix("-rgb.png")
        source_ids_path = render_dir / f"{stem}-source-ids.npy"
        if not source_ids_path.is_file():
            raise FileNotFoundError(source_ids_path)
        image = Image.open(image_path).convert("RGB")
        labels, id2label = predictor.predict(image)
        source_ids = np.load(source_ids_path)
        if labels.shape != source_ids.shape:
            raise ValueError(
                f"semantic labels for {stem} do not match source-ID render"
            )
        valid = source_ids >= 0
        visible_ids = source_ids[valid].astype(
            source_indices.dtype,
            copy=False,
        )
        rows = np.searchsorted(source_indices, visible_ids)
        if np.any(rows >= len(source_indices)) or np.any(
            source_indices[rows] != visible_ids
        ):
            raise ValueError(f"{stem} references unknown source IDs")
        visible_labels = labels[valid]
        plant_ids = [
            label_id
            for label_id, name in id2label.items()
            if name in plant_labels
        ]
        background_ids = (
            [
                label_id
                for label_id in id2label
                if label_id not in plant_ids
            ]
            if background_labels == ("*",)
            else [
                label_id
                for label_id, name in id2label.items()
                if name in background_labels
            ]
        )
        plant = np.isin(visible_labels, plant_ids)
        background = np.isin(visible_labels, background_ids)
        np.add.at(plant_votes, rows[plant], 1)
        np.add.at(background_votes, rows[background], 1)
        view_reports.append(
            {
                "view": stem,
                "visible_pixel_count": int(valid.sum()),
                "plant_pixel_count": int(plant.sum()),
                "background_pixel_count": int(background.sum()),
            }
        )

    if not view_reports:
        raise ValueError("dense semantic evidence requires orbit renders")
    np.save(output_dir / "plant-votes.npy", plant_votes)
    np.save(output_dir / "background-votes.npy", background_votes)
    report = {
        "schema_version": 1,
        "model": model_id,
        "plant_labels": list(plant_labels),
        "background_labels": list(background_labels),
        "point_count": int(len(cloud)),
        "plant_seed_count": int(np.count_nonzero(plant_votes > background_votes)),
        "background_seed_count": int(
            np.count_nonzero(background_votes > plant_votes)
        ),
        "views": view_reports,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


@dataclass(frozen=True)
class DensePropagationParameters:
    conservative_background_factor: float = 2.0
    strict_background_factor: float = 10.0
    ground_height_fraction: float = 0.05
    ground_normal_min: float = 0.60
    structural_protection_radius_fraction: float = 0.05
    minimum_seed_points: int = 50


def propagate_dense_semantic_evidence(
    coordinates: np.ndarray,
    *,
    normals: np.ndarray,
    candidate_mask: np.ndarray,
    plant_votes: np.ndarray,
    background_votes: np.ndarray,
    support_plane_coefficients: tuple[float, float, float],
    vertical_span: float,
    parameters: DensePropagationParameters | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Turn sparse 2D semantic votes into competing 3D distance fields."""

    parameters = parameters or DensePropagationParameters()
    coordinates = np.asarray(coordinates, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    plant_votes = np.asarray(plant_votes)
    background_votes = np.asarray(background_votes)
    point_count = len(coordinates)
    if coordinates.shape != (point_count, 3):
        raise ValueError("coordinates must have shape (point_count, 3)")
    if normals.shape != coordinates.shape:
        raise ValueError("normals must match coordinates")
    for name, values in (
        ("candidate_mask", candidate_mask),
        ("plant_votes", plant_votes),
        ("background_votes", background_votes),
    ):
        if values.shape != (point_count,):
            raise ValueError(f"{name} must have shape ({point_count},)")
    if vertical_span <= 0:
        raise ValueError("vertical_span must be positive")

    a, b, c = support_plane_coefficients
    plane_height = coordinates[:, 2] - (
        a * coordinates[:, 0] + b * coordinates[:, 1] + c
    )
    ground_like = (
        plane_height
        < vertical_span * parameters.ground_height_fraction
    ) & (np.abs(normals[:, 2]) >= parameters.ground_normal_min)
    model_plant = plant_votes > background_votes
    model_background = background_votes > plant_votes
    plant_seed = model_plant & ~ground_like
    background_seed = model_background | (model_plant & ground_like)

    if (
        int(plant_seed.sum()) < parameters.minimum_seed_points
        or int(background_seed.sum()) < parameters.minimum_seed_points
    ):
        report = {
            "schema_version": 1,
            "status": "insufficient_evidence",
            "parameters": asdict(parameters),
            "plant_seed_count": int(plant_seed.sum()),
            "background_seed_count": int(background_seed.sum()),
            "candidate_point_count": int(candidate_mask.sum()),
            "strict_point_count": int(candidate_mask.sum()),
            "conservative_point_count": int(candidate_mask.sum()),
        }
        return candidate_mask.copy(), candidate_mask.copy(), report

    plant_distance = cKDTree(coordinates[plant_seed]).query(
        coordinates,
        k=1,
        workers=-1,
    )[0]
    background_distance = cKDTree(coordinates[background_seed]).query(
        coordinates,
        k=1,
        workers=-1,
    )[0]
    structural_protected = (
        ~ground_like
        & (np.abs(normals[:, 2]) < parameters.ground_normal_min)
        & (
            plant_distance
            <= (
                vertical_span
                * parameters.structural_protection_radius_fraction
            )
        )
        & (background_votes <= plant_votes)
    )
    protected = (model_plant & ~ground_like) | structural_protected

    def keep_for_factor(factor: float) -> np.ndarray:
        background = (
            background_distance < plant_distance * factor
        ) & ~protected
        return candidate_mask & ~background

    conservative = keep_for_factor(
        parameters.conservative_background_factor
    )
    strict = keep_for_factor(parameters.strict_background_factor)
    report = {
        "schema_version": 1,
        "status": "complete",
        "parameters": asdict(parameters),
        "plant_seed_count": int(plant_seed.sum()),
        "background_seed_count": int(background_seed.sum()),
        "ground_reclassified_seed_count": int(
            np.count_nonzero(model_plant & ground_like)
        ),
        "structural_protected_point_count": int(
            structural_protected.sum()
        ),
        "candidate_point_count": int(candidate_mask.sum()),
        "strict_point_count": int(strict.sum()),
        "conservative_point_count": int(conservative.sum()),
    }
    return strict, conservative, report
