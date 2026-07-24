from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from plant_cleanup.cloud_render import BACKGROUND
from plant_cleanup.plyio import read_cloud


Predictor = Callable[
    [Image.Image, list[list[float]], list[int]], tuple[np.ndarray, float]
]


def _select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    image: np.ndarray,
    points: list[list[float]],
    labels: list[int],
) -> int:
    """Prefer masks that obey prompts and do not select the render background."""
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError("SAM2 masks and scores must have matching candidates")
    background = np.all(np.asarray(image) == BACKGROUND, axis=2)
    objectives: list[float] = []
    diagnostics: list[dict[str, float]] = []
    for mask, score in zip(masks, scores, strict=True):
        positive_hits = [
            mask[int(y), int(x)]
            for (x, y), label in zip(points, labels, strict=True)
            if label == 1
        ]
        negative_hits = [
            mask[int(y), int(x)]
            for (x, y), label in zip(points, labels, strict=True)
            if label == 0
        ]
        positive_fraction = float(np.mean(positive_hits))
        negative_fraction = float(np.mean(negative_hits))
        background_fraction = float(mask[background].mean()) if background.any() else 0.0
        prompt_valid = positive_fraction >= 0.5 and negative_fraction <= 0.25
        background_valid = background_fraction <= 0.25
        objectives.append(float(score) if prompt_valid and background_valid else -np.inf)
        diagnostics.append(
            {
                "score": float(score),
                "positive_fraction": positive_fraction,
                "negative_fraction": negative_fraction,
                "background_fraction": background_fraction,
            }
        )
    if not np.isfinite(objectives).any():
        raise ValueError(
            f"SAM2 produced no prompt-consistent foreground mask: {diagnostics}"
        )
    return int(np.argmax(objectives))


class HuggingFaceSam2Predictor:
    """Promptable SAM2 image segmentation with one cached model instance."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).eval().to(self.device)

    def __call__(
        self, image: Image.Image, points: list[list[float]], labels: list[int]
    ) -> tuple[np.ndarray, float]:
        inputs = self.processor(
            images=image,
            input_points=[[points]],
            input_labels=[[labels]],
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0][0]
        scores = outputs.iou_scores[0, 0].detach().cpu().numpy()
        best = _select_best_mask(
            masks.numpy(), scores, np.asarray(image), points, labels
        )
        return masks[best].numpy().astype(bool), float(scores[best])


def _artifact_path(parent: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return parent / path


def _sample_grid_maxima(
    score: np.ndarray,
    candidate: np.ndarray,
    grid: tuple[int, int],
) -> list[list[float]]:
    rows, columns = grid
    if rows < 1 or columns < 1:
        raise ValueError("prompt grid dimensions must be positive")
    height, width = score.shape
    points: list[list[float]] = []
    for row in range(rows):
        y0, y1 = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            x0, x1 = column * width // columns, (column + 1) * width // columns
            local = np.where(
                candidate[y0:y1, x0:x1], score[y0:y1, x0:x1], -np.inf
            )
            flat_index = int(np.argmax(local))
            if np.isfinite(local.flat[flat_index]):
                y, x = np.unravel_index(flat_index, local.shape)
                points.append([float(x + x0), float(y + y0)])
    return points


def _project_coordinates_for_view(
    coordinates: np.ndarray,
    render_report: dict[str, Any],
    view_name: str,
) -> np.ndarray:
    frame = render_report["frame"]
    center = np.asarray(frame["center"], dtype=np.float64)
    axes = np.asarray(frame["axes"], dtype=np.float64)
    canonical = (np.asarray(coordinates, dtype=np.float64) - center) @ axes
    if view_name == "front":
        return canonical[:, (0, 2, 1)]
    if view_name == "side":
        return canonical[:, (1, 2, 0)]
    if view_name == "top":
        return canonical[:, (0, 1, 2)]
    if view_name.startswith("orbit-"):
        degrees = int(view_name.removeprefix("orbit-"))
        radians = np.deg2rad(degrees)
        cosine = float(np.cos(radians))
        sine = float(np.sin(radians))
        horizontal = cosine * canonical[:, 0] + sine * canonical[:, 1]
        depth = -sine * canonical[:, 0] + cosine * canonical[:, 1]
        return np.column_stack((horizontal, canonical[:, 2], depth))
    raise ValueError(f"unsupported render view for depth projection: {view_name!r}")


def _depth_complete_mask(
    coordinates: np.ndarray,
    mask: np.ndarray,
    depth_buffer: np.ndarray,
    render_report: dict[str, Any],
    view: dict[str, Any],
    *,
    depth_fraction: float,
) -> np.ndarray:
    selected = np.zeros(len(coordinates), dtype=bool)
    projection = view.get("projection")
    if depth_fraction <= 0 or not projection or "frame" not in render_report:
        return selected
    projected = _project_coordinates_for_view(
        coordinates, render_report, view["view"]
    )
    size = int(projection["size"])
    scale = float(projection["scale"])
    x = np.rint(
        (projected[:, 0] - float(projection["horizontal_center"])) * scale
        + (size - 1) / 2
    ).astype(np.int32)
    y = np.rint(
        (float(projection["vertical_center"]) - projected[:, 1]) * scale
        + (size - 1) / 2
    ).astype(np.int32)
    in_frame = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    rows = np.flatnonzero(in_frame)
    pixel_depth = depth_buffer[y[rows], x[rows]]
    depth_tolerance = float(projection["scene_extent"]) * depth_fraction
    depth_close = np.isfinite(pixel_depth) & (
        np.abs(projected[rows, 2] - pixel_depth) <= depth_tolerance
    )
    rows = rows[depth_close]
    selected[rows[mask[y[rows], x[rows]]]] = True
    return selected


def _spread_anchors(
    points: list[list[float]], score: np.ndarray, limit: int = 3
) -> list[list[float]]:
    """Choose a few high-scoring, spatially distributed object anchors."""
    if len(points) <= limit:
        return points
    remaining = list(points)
    first = max(remaining, key=lambda point: float(score[int(point[1]), int(point[0])]))
    selected = [first]
    remaining.remove(first)
    height, width = score.shape
    while remaining and len(selected) < limit:
        def objective(point: list[float]) -> tuple[float, float]:
            distance = min(
                ((point[0] - other[0]) / max(width, 1)) ** 2
                + ((point[1] - other[1]) / max(height, 1)) ** 2
                for other in selected
            )
            return distance, float(score[int(point[1]), int(point[0])])

        chosen = max(remaining, key=objective)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _predict_anchor_union(
    predictor: Predictor,
    image: Image.Image,
    positive: list[list[float]],
    negative: list[list[float]],
) -> tuple[np.ndarray, float, int]:
    """Segment disconnected objects independently, then union only valid masks."""
    union = np.zeros((image.height, image.width), dtype=bool)
    scores: list[float] = []
    failures: list[str] = []
    for anchor in positive:
        try:
            mask, score = predictor(
                image,
                [anchor] + negative,
                [1] + [0] * len(negative),
            )
        except ValueError as error:
            failures.append(str(error))
            continue
        union |= np.asarray(mask, dtype=bool)
        scores.append(float(score))
    if not scores:
        raise ValueError(f"all independent SAM2 anchors failed: {failures}")
    return union, float(np.mean(scores)), len(scores)


def aggregate_sam2_votes(
    cloud_path: Path,
    render_dir: Path,
    clipseg_dir: Path,
    output_dir: Path,
    *,
    predictor: Predictor | None = None,
    model_id: str = "facebook/sam2-hiera-tiny",
    prompt_grid: tuple[int, int] = (3, 6),
    margin_min: float = 0.08,
    geometry_decisions: np.ndarray | None = None,
    ground_prompt_mask: np.ndarray | None = None,
    background_depth_fraction: float = 0.0,
    background_anchor_limit: int = 3,
) -> dict[str, Any]:
    """Refine coarse text masks with SAM2 and project competing 3D votes."""
    cloud_path = cloud_path.resolve()
    render_dir = render_dir.resolve()
    clipseg_dir = clipseg_dir.resolve()
    output_dir = output_dir.resolve()
    cloud = read_cloud(cloud_path)
    source_indices = np.asarray(cloud["source_index"])
    if len(source_indices) > 1 and np.any(source_indices[1:] <= source_indices[:-1]):
        raise ValueError("source_index must be strictly increasing")
    if geometry_decisions is not None:
        geometry_decisions = np.asarray(geometry_decisions, dtype=np.uint8)
        if len(geometry_decisions) != len(cloud):
            raise ValueError("geometry decisions must match cloud length")
    if ground_prompt_mask is not None:
        ground_prompt_mask = np.asarray(ground_prompt_mask, dtype=bool)
        if len(ground_prompt_mask) != len(cloud):
            raise ValueError("ground prompt mask must match cloud length")
    if background_depth_fraction < 0:
        raise ValueError("background depth fraction must be nonnegative")
    if not 1 <= background_anchor_limit <= 32:
        raise ValueError("background anchor limit must be between 1 and 32")
    coordinates = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(
        np.float64, copy=False
    )
    render_report = json.loads((render_dir / "render-report.json").read_text())
    predictor = predictor or HuggingFaceSam2Predictor(model_id)
    plant_votes = np.zeros(len(cloud), dtype=np.uint8)
    planter_votes = np.zeros(len(cloud), dtype=np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    view_reports: list[dict[str, Any]] = []

    for view in render_report["views"]:
        name = view["view"]
        image = Image.open(_artifact_path(render_dir, view["rgb"])).convert("RGB")
        probabilities = np.load(clipseg_dir / f"{name}-probabilities.npy")
        expected_shape = (3, image.height, image.width)
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"{name} probabilities are {probabilities.shape}, expected {expected_shape}"
            )
        id_buffer = np.load(_artifact_path(render_dir, view["source_ids"]))
        visible = id_buffer >= 0
        visible_ids = id_buffer[visible]
        visible_rows = np.searchsorted(source_indices, visible_ids)
        if np.any(visible_rows >= len(source_indices)) or not np.array_equal(
            source_indices[visible_rows], visible_ids
        ):
            raise ValueError(f"{name} contains IDs absent from the source cloud")
        plant_margin = probabilities[0] - np.maximum(
            probabilities[1], probabilities[2]
        )
        planter_margin = probabilities[1] - np.maximum(
            probabilities[0], probabilities[2]
        )
        semantic_positive = _sample_grid_maxima(
            planter_margin, visible & (planter_margin > margin_min), prompt_grid
        )
        semantic_negative = _sample_grid_maxima(
            plant_margin, visible & (plant_margin > margin_min), prompt_grid
        )
        positive = semantic_positive
        negative = semantic_negative
        planter_prompt_source = "clipseg_margin"
        plant_prompt_source = "clipseg_margin"
        if geometry_decisions is not None:
            support_pixels = np.zeros_like(visible)
            plant_pixels = np.zeros_like(visible)
            support_pixels[visible] = geometry_decisions[visible_rows] == 2
            if ground_prompt_mask is not None:
                support_pixels[visible] |= ground_prompt_mask[visible_rows]
            plant_pixels[visible] = geometry_decisions[visible_rows] == 1
            geometry_positive = _sample_grid_maxima(
                planter_margin, support_pixels, prompt_grid
            )
            geometry_negative = _sample_grid_maxima(
                plant_margin, plant_pixels, prompt_grid
            )
            if geometry_positive:
                positive = geometry_positive
                planter_prompt_source = (
                    "geometry_support_and_fitted_ground_ranked_by_clipseg"
                    if ground_prompt_mask is not None
                    else "geometry_support_ranked_by_clipseg"
                )
            if not negative and geometry_negative:
                negative = geometry_negative
                plant_prompt_source = "geometry_plant_ranked_by_clipseg"
        if not positive or not negative:
            missing = []
            if not positive:
                missing.append("planter")
            if not negative:
                missing.append("plant")
            view_reports.append(
                {
                    "view": name,
                    "status": "object_not_visible",
                    "missing_prompt_classes": missing,
                }
            )
            continue
        planter_anchors = _spread_anchors(
            positive, planter_margin, limit=background_anchor_limit
        )
        plant_anchors = _spread_anchors(negative, plant_margin)
        planter_mask, planter_iou, planter_successes = _predict_anchor_union(
            predictor, image, planter_anchors, plant_anchors
        )
        plant_mask, plant_iou, plant_successes = _predict_anchor_union(
            predictor, image, plant_anchors, planter_anchors
        )
        planter_mask = np.asarray(planter_mask, dtype=bool)
        plant_mask = np.asarray(plant_mask, dtype=bool)
        expected_mask_shape = (image.height, image.width)
        if planter_mask.shape != expected_mask_shape or plant_mask.shape != expected_mask_shape:
            raise ValueError(
                f"{name} predictor returned plant {plant_mask.shape} and "
                f"planter {planter_mask.shape}, expected {expected_mask_shape}"
            )

        Image.fromarray((planter_mask * 255).astype(np.uint8)).save(
            output_dir / f"{name}-planter-mask.png"
        )
        Image.fromarray((plant_mask * 255).astype(np.uint8)).save(
            output_dir / f"{name}-plant-mask.png"
        )
        rgb = np.asarray(image).copy()
        overlay = rgb.copy()
        overlay[plant_mask] = (40, 255, 80)
        overlay[planter_mask] = (255, 55, 40)
        points = planter_anchors + plant_anchors
        labels = [1] * len(planter_anchors) + [0] * len(plant_anchors)
        for (x, y), label in zip(points, labels, strict=True):
            color = (255, 255, 0) if label == 1 else (0, 255, 255)
            x_int, y_int = int(x), int(y)
            overlay[
                max(0, y_int - 4) : y_int + 5,
                max(0, x_int - 4) : x_int + 5,
            ] = color
        Image.fromarray((rgb * 0.45 + overlay * 0.55).astype(np.uint8)).save(
            output_dir / f"{name}-overlay.png"
        )

        unique_ids, inverse = np.unique(visible_ids, return_inverse=True)
        totals = np.bincount(inverse)
        rows = np.searchsorted(source_indices, unique_ids)
        if np.any(rows >= len(source_indices)) or not np.array_equal(
            source_indices[rows], unique_ids
        ):
            raise ValueError(f"{name} contains IDs absent from the source cloud")
        def project(mask: np.ndarray) -> np.ndarray:
            positive_pixels = np.bincount(
                inverse, weights=mask[visible].astype(np.uint8)
            )
            return positive_pixels * 2 >= totals

        chosen_planter = project(planter_mask)
        chosen_plant = project(plant_mask)
        chosen_planter_rows = np.zeros(len(cloud), dtype=bool)
        chosen_planter_rows[rows[chosen_planter]] = True
        if (
            background_depth_fraction > 0
            and "depth" in view
            and geometry_decisions is not None
        ):
            depth_buffer = np.load(_artifact_path(render_dir, view["depth"]))
            if depth_buffer.shape != expected_mask_shape:
                raise ValueError(
                    f"{name} depth buffer is {depth_buffer.shape}, "
                    f"expected {expected_mask_shape}"
                )
            depth_completed = _depth_complete_mask(
                coordinates,
                planter_mask,
                depth_buffer,
                render_report,
                view,
                depth_fraction=background_depth_fraction,
            )
            # Never infer an occluded man-made/background point from depth
            # alone. Completion may only reinforce points already classified
            # as support by geometry; plant candidates, roots, and uncertain
            # structure remain untouched.
            depth_completed &= geometry_decisions == 2
            depth_completed &= ~chosen_planter_rows
            chosen_planter_rows |= depth_completed
        else:
            depth_completed = np.zeros(len(cloud), dtype=bool)
        planter_votes[chosen_planter_rows] += 1
        plant_votes[rows[chosen_plant]] += 1
        view_reports.append(
            {
                "view": name,
                "status": "segmented",
                "positive_prompt_count": len(planter_anchors),
                "negative_prompt_count": len(plant_anchors),
                "planter_successful_anchor_count": planter_successes,
                "plant_successful_anchor_count": plant_successes,
                "planter_prompt_source": planter_prompt_source,
                "plant_prompt_source": plant_prompt_source,
                "planter_predicted_iou": planter_iou,
                "plant_predicted_iou": plant_iou,
                "planter_point_ids": int(chosen_planter_rows.sum()),
                "depth_completed_planter_points": int(depth_completed.sum()),
                "plant_point_ids": int(chosen_plant.sum()),
            }
        )

    np.save(output_dir / "planter-votes.npy", planter_votes)
    np.save(output_dir / "plant-votes.npy", plant_votes)
    report = {
        "cloud": str(cloud_path),
        "render_report": str((render_dir / "render-report.json").resolve()),
        "clipseg_dir": str(clipseg_dir),
        "model": model_id,
        "prompt_source": "automatic CLIPSeg class-margin grid maxima",
        "prompt_grid": list(prompt_grid),
        "margin_min": margin_min,
        "background_depth_fraction": background_depth_fraction,
        "depth_completion_policy": "geometry_support_only",
        "background_anchor_limit": background_anchor_limit,
        "views": view_reports,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


# Compatibility for the initial single-class prototype name.
aggregate_sam2_planter_votes = aggregate_sam2_votes
