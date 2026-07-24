from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from plant_cleanup.plyio import read_cloud


DEFAULT_PROMPTS = (
    "plants and leaves",
    "concrete planter wall or pavement",
    "floating reconstruction noise",
)

Predictor = Callable[[Image.Image, tuple[str, ...]], np.ndarray]


class HuggingFaceClipSegPredictor:
    def __init__(self, model_id: str, device: str | None = None) -> None:
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(model_id, use_fast=False)
        self.model = (
            CLIPSegForImageSegmentation.from_pretrained(model_id)
            .eval()
            .to(self.device)
        )

    def __call__(self, image: Image.Image, prompts: tuple[str, ...]) -> np.ndarray:
        inputs = self.processor(
            text=list(prompts),
            images=[image] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        resized = self.torch.nn.functional.interpolate(
            logits[:, None],
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return self.torch.sigmoid(resized).cpu().numpy()


def _artifact_path(render_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return render_dir / path


def aggregate_clipseg_votes(
    cloud_path: Path,
    render_dir: Path,
    output_dir: Path,
    *,
    predictor: Predictor | None = None,
    model_id: str = "CIDAS/clipseg-rd64-refined",
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    confidence_min: float = 0.25,
    margin_min: float = 0.02,
) -> dict[str, Any]:
    """Project multi-prompt CLIPSeg masks back to stable source point IDs."""
    cloud_path = cloud_path.resolve()
    render_dir = render_dir.resolve()
    output_dir = output_dir.resolve()
    if len(prompts) != 3:
        raise ValueError("exactly plant, planter, and noise prompts are required")
    cloud = read_cloud(cloud_path)
    source_indices = np.asarray(cloud["source_index"])
    if len(source_indices) > 1 and np.any(source_indices[1:] <= source_indices[:-1]):
        raise ValueError("source_index must be strictly increasing")
    render_report = json.loads((render_dir / "render-report.json").read_text())
    predictor = predictor or HuggingFaceClipSegPredictor(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    vote_arrays = [np.zeros(len(cloud), dtype=np.uint8) for _ in prompts]
    view_reports: list[dict[str, Any]] = []

    for view in render_report["views"]:
        name = view["view"]
        image = Image.open(_artifact_path(render_dir, view["rgb"])).convert("RGB")
        probabilities = np.asarray(predictor(image, prompts), dtype=np.float32)
        expected_shape = (3, image.height, image.width)
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"predictor returned {probabilities.shape}, expected {expected_shape}"
            )
        np.save(output_dir / f"{name}-probabilities.npy", probabilities)
        winner = probabilities.argmax(axis=0)
        highest = probabilities.max(axis=0)
        second = np.partition(probabilities, -2, axis=0)[-2]
        confident = (highest > confidence_min) & (highest - second > margin_min)
        masks = [(winner == index) & confident for index in range(3)]

        Image.fromarray((masks[0] * 255).astype(np.uint8)).save(
            output_dir / f"{name}-plant-mask.png"
        )
        base = np.asarray(image).copy()
        overlay = base.copy()
        overlay[masks[0]] = (40, 255, 80)
        overlay[masks[1]] = (255, 60, 40)
        Image.fromarray((base * 0.45 + overlay * 0.55).astype(np.uint8)).save(
            output_dir / f"{name}-overlay.png"
        )

        id_buffer = np.load(_artifact_path(render_dir, view["source_ids"]))
        visible = id_buffer >= 0
        visible_ids = id_buffer[visible]
        unique_ids, inverse = np.unique(visible_ids, return_inverse=True)
        totals = np.bincount(inverse)
        rows = np.searchsorted(source_indices, unique_ids)
        if np.any(rows >= len(source_indices)) or not np.array_equal(
            source_indices[rows], unique_ids
        ):
            raise ValueError(f"{name} contains IDs absent from the source cloud")

        chosen_counts: list[int] = []
        for mask, votes in zip(masks, vote_arrays, strict=True):
            positive = np.bincount(
                inverse, weights=mask[visible].astype(np.uint8)
            )
            chosen = positive * 2 >= totals
            votes[rows[chosen]] += 1
            chosen_counts.append(int(chosen.sum()))
        view_reports.append(
            {
                "view": name,
                "visible_point_ids": int(len(unique_ids)),
                "plant_point_ids": chosen_counts[0],
                "planter_point_ids": chosen_counts[1],
                "noise_point_ids": chosen_counts[2],
            }
        )

    for name, votes in zip(
        ("plant-votes", "planter-votes", "noise-votes"),
        vote_arrays,
        strict=True,
    ):
        np.save(output_dir / f"{name}.npy", votes)
    report = {
        "cloud": str(cloud_path),
        "render_report": str((render_dir / "render-report.json").resolve()),
        "model": model_id,
        "prompts": list(prompts),
        "confidence_min": confidence_min,
        "margin_min": margin_min,
        "views": view_reports,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
