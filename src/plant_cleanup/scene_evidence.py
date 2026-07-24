from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from plant_cleanup.clipseg_votes import Predictor as ClipSegPredictor
from plant_cleanup.clipseg_votes import aggregate_clipseg_votes
from plant_cleanup.sam2_votes import Predictor as Sam2Predictor
from plant_cleanup.sam2_votes import aggregate_sam2_votes


CLASS_ID = re.compile(r"^[a-z][a-z0-9_-]*$")


def _create_scene_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError as error:
        raise FileExistsError(
            f"scene evidence output already exists: {output_dir}"
        ) from error
    return output_dir


def fuse_scene_votes(
    class_runs: Mapping[str, Path],
    output_dir: Path,
    *,
    class_policies: Mapping[str, str] | None = None,
) -> dict:
    """Fuse named non-plant object evidence without losing class identity."""
    if not class_runs:
        raise ValueError("at least one scene evidence class is required")
    output_dir = _create_scene_output_dir(output_dir)
    policies = dict(class_policies or {})
    if set(policies) - set(class_runs):
        raise ValueError("scene class policies must match evidence classes")
    plant_arrays: list[np.ndarray] = []
    background_arrays: list[np.ndarray] = []
    class_report: dict[str, dict[str, int]] = {}
    point_count: int | None = None

    for class_id, run_dir in class_runs.items():
        if not CLASS_ID.fullmatch(class_id):
            raise ValueError(f"invalid scene evidence class id: {class_id!r}")
        run_dir = run_dir.resolve()
        plant = np.asarray(np.load(run_dir / "plant-votes.npy"), dtype=np.uint8)
        background = np.asarray(
            np.load(run_dir / "planter-votes.npy"), dtype=np.uint8
        )
        if plant.ndim != 1 or background.ndim != 1 or plant.shape != background.shape:
            raise ValueError(f"{class_id} vote arrays must be matching vectors")
        if point_count is None:
            point_count = len(plant)
        elif len(plant) != point_count:
            raise ValueError("scene evidence classes must cover the same source points")
        plant_arrays.append(plant)
        background_arrays.append(background)
        np.save(output_dir / f"{class_id}-votes.npy", background)
        np.save(output_dir / f"{class_id}-plant-votes.npy", plant)
        class_report[class_id] = {
            "voted_point_count": int(np.count_nonzero(background)),
            "maximum_view_votes": int(background.max(initial=0)),
        }

    plant_votes = np.maximum.reduce(plant_arrays)
    background_votes = np.maximum.reduce(background_arrays)
    np.save(output_dir / "plant-votes.npy", plant_votes)
    np.save(output_dir / "background-votes.npy", background_votes)
    if policies:
        (output_dir / "class-policies.json").write_text(
            json.dumps(policies, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = {
        "point_count": int(point_count or 0),
        "plant_voted_point_count": int(np.count_nonzero(plant_votes)),
        "background_voted_point_count": int(np.count_nonzero(background_votes)),
        "conflicting_point_count": int(
            np.count_nonzero((plant_votes > 0) & (background_votes > 0))
        ),
        "classes": class_report,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def merge_scene_evidence(
    iteration_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Merge initial and residual scene evidence without double-counting views."""
    if not iteration_dirs:
        raise ValueError("at least one scene evidence iteration is required")
    output_dir = _create_scene_output_dir(output_dir)
    plant_arrays: list[np.ndarray] = []
    class_arrays: dict[str, list[np.ndarray]] = {}
    class_plant_arrays: dict[str, list[np.ndarray]] = {}
    class_policies: dict[str, str] = {}
    point_count: int | None = None

    for iteration_dir in iteration_dirs:
        iteration_dir = iteration_dir.resolve()
        plant = np.asarray(
            np.load(iteration_dir / "plant-votes.npy"), dtype=np.uint8
        )
        if plant.ndim != 1:
            raise ValueError("scene plant votes must be a vector")
        if point_count is None:
            point_count = len(plant)
        elif len(plant) != point_count:
            raise ValueError("scene evidence iterations must cover the same points")
        plant_arrays.append(plant)
        policy_path = iteration_dir / "class-policies.json"
        iteration_policies = (
            json.loads(policy_path.read_text(encoding="utf-8"))
            if policy_path.exists()
            else {}
        )
        for class_id, policy in iteration_policies.items():
            previous = class_policies.setdefault(class_id, policy)
            if previous != policy:
                raise ValueError(
                    f"scene class {class_id!r} changed policy across iterations"
                )
        for path in iteration_dir.glob("*-votes.npy"):
            if path.name.endswith("-plant-votes.npy"):
                class_id = path.name.removesuffix("-plant-votes.npy")
                votes = np.asarray(np.load(path), dtype=np.uint8)
                if votes.shape != plant.shape:
                    raise ValueError(
                        f"{class_id} paired plant votes do not match plant votes"
                    )
                class_plant_arrays.setdefault(class_id, []).append(votes)
                continue
            class_id = path.name.removesuffix("-votes.npy")
            if class_id in {"plant", "background"}:
                continue
            votes = np.asarray(np.load(path), dtype=np.uint8)
            if votes.shape != plant.shape:
                raise ValueError(f"{class_id} votes do not match plant votes")
            class_arrays.setdefault(class_id, []).append(votes)

    if not class_arrays:
        raise ValueError("scene evidence iterations contain no object classes")
    plant_votes = np.maximum.reduce(plant_arrays)
    merged_classes = {
        class_id: np.maximum.reduce(arrays)
        for class_id, arrays in class_arrays.items()
    }
    merged_class_plants = {
        class_id: np.maximum.reduce(arrays)
        for class_id, arrays in class_plant_arrays.items()
    }
    background_votes = np.maximum.reduce(list(merged_classes.values()))
    np.save(output_dir / "plant-votes.npy", plant_votes)
    np.save(output_dir / "background-votes.npy", background_votes)
    for class_id, votes in merged_classes.items():
        np.save(output_dir / f"{class_id}-votes.npy", votes)
    for class_id, votes in merged_class_plants.items():
        np.save(output_dir / f"{class_id}-plant-votes.npy", votes)
    if class_policies:
        (output_dir / "class-policies.json").write_text(
            json.dumps(class_policies, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report = {
        "iteration_count": len(iteration_dirs),
        "iterations": [str(path.resolve()) for path in iteration_dirs],
        "point_count": int(point_count or 0),
        "plant_voted_point_count": int(np.count_nonzero(plant_votes)),
        "background_voted_point_count": int(np.count_nonzero(background_votes)),
        "conflicting_point_count": int(
            np.count_nonzero((plant_votes > 0) & (background_votes > 0))
        ),
        "classes": {
            class_id: {
                "voted_point_count": int(np.count_nonzero(votes)),
                "maximum_view_votes": int(votes.max(initial=0)),
                "paired_plant_voted_point_count": int(
                    np.count_nonzero(
                        merged_class_plants.get(
                            class_id, np.zeros_like(votes)
                        )
                    )
                ),
            }
            for class_id, votes in merged_classes.items()
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_scene_evidence(
    cloud_path: Path,
    render_dir: Path,
    output_dir: Path,
    plan: Mapping[str, Any],
    *,
    clipseg_predictor: ClipSegPredictor | None = None,
    sam2_predictor: Sam2Predictor | None = None,
    clipseg_model: str = "CIDAS/clipseg-rd64-refined",
    sam2_model: str = "facebook/sam2-hiera-tiny",
    confidence_min: float = 0.2,
    clipseg_margin_min: float = 0.01,
    sam2_margin_min: float = 0.05,
    prompt_grid: tuple[int, int] = (3, 6),
    geometry_decisions: np.ndarray | None = None,
    ground_prompt_mask: np.ndarray | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run each planned non-plant object as an independent vision pass."""
    if plan.get("schema_version") != 1:
        raise ValueError("scene plan schema_version must be 1")
    scan_id = str(plan.get("scan_id", "")).strip()
    plant_prompt = str(plan.get("plant_prompt", "")).strip()
    classes = plan.get("classes")
    if not scan_id or not plant_prompt or not isinstance(classes, list) or not classes:
        raise ValueError("scene plan requires scan_id, plant_prompt, and classes")

    output_dir = _create_scene_output_dir(output_dir)
    class_runs: dict[str, Path] = {}
    class_reports: dict[str, dict[str, Any]] = {}
    class_policies: dict[str, str] = {}
    for object_class in classes:
        class_id = str(object_class.get("id", ""))
        if not CLASS_ID.fullmatch(class_id):
            raise ValueError(f"invalid scene evidence class id: {class_id!r}")
        prompt = str(object_class.get("prompt", "")).strip()
        distractor = str(object_class.get("distractor_prompt", "")).strip()
        strategy = object_class.get("anchor_strategy", "semantic")
        decision_policy = object_class.get(
            "decision_policy",
            "ground_surface" if strategy == "geometry_ground" else "strict_background",
        )
        background_depth_fraction = float(
            object_class.get("background_depth_fraction", 0.0)
        )
        background_anchor_limit = int(
            object_class.get("background_anchor_limit", 3)
        )
        required_segmented_views = int(
            object_class.get("required_segmented_views", 1)
        )
        if not prompt or not distractor:
            raise ValueError(f"{class_id} requires prompt and distractor_prompt")
        if strategy not in {"semantic", "geometry_ground"}:
            raise ValueError(f"{class_id} has unsupported anchor_strategy {strategy!r}")
        if decision_policy not in {
            "strict_background",
            "ground_surface",
            "class_exclusive_background",
        }:
            raise ValueError(
                f"{class_id} has unsupported decision_policy {decision_policy!r}"
            )
        if not 0 <= background_depth_fraction <= 0.25:
            raise ValueError(
                f"{class_id} background_depth_fraction must be between 0 and 0.25"
            )
        if not 1 <= background_anchor_limit <= 32:
            raise ValueError(
                f"{class_id} background_anchor_limit must be between 1 and 32"
            )
        if not 1 <= required_segmented_views <= 32:
            raise ValueError(
                f"{class_id} required_segmented_views must be between 1 and 32"
            )
        if strategy == "geometry_ground" and geometry_decisions is None:
            raise ValueError(f"{class_id} requires geometry decisions")

        class_dir = output_dir / "classes" / class_id
        clipseg_dir = class_dir / "clipseg"
        sam2_dir = class_dir / "sam2"
        clipseg_report = aggregate_clipseg_votes(
            cloud_path,
            render_dir,
            clipseg_dir,
            predictor=clipseg_predictor,
            model_id=clipseg_model,
            prompts=(plant_prompt, prompt, distractor),
            confidence_min=confidence_min,
            margin_min=clipseg_margin_min,
        )
        sam2_report = aggregate_sam2_votes(
            cloud_path,
            render_dir,
            clipseg_dir,
            sam2_dir,
            predictor=sam2_predictor,
            model_id=sam2_model,
            prompt_grid=prompt_grid,
            margin_min=sam2_margin_min,
            geometry_decisions=(
                geometry_decisions if strategy == "geometry_ground" else None
            ),
            ground_prompt_mask=(
                ground_prompt_mask if strategy == "geometry_ground" else None
            ),
            background_depth_fraction=background_depth_fraction,
            background_anchor_limit=background_anchor_limit,
        )
        segmented_views = sum(
            view.get("status") == "segmented"
            for view in sam2_report["views"]
        )
        quality_state = (
            "usable"
            if segmented_views >= required_segmented_views
            else "insufficient_segmented_views"
        )
        class_runs[class_id] = sam2_dir
        class_policies[class_id] = decision_policy
        class_reports[class_id] = {
            "prompt": prompt,
            "distractor_prompt": distractor,
            "anchor_strategy": strategy,
            "decision_policy": decision_policy,
            "background_depth_fraction": background_depth_fraction,
            "background_anchor_limit": background_anchor_limit,
            "required_segmented_views": required_segmented_views,
            "segmented_views": segmented_views,
            "quality_state": quality_state,
            "clipseg": clipseg_report,
            "sam2": sam2_report,
        }

    fused = fuse_scene_votes(
        class_runs,
        output_dir / "fused",
        class_policies=class_policies,
    )
    report = {
        "scan_id": scan_id,
        "target_intent": str(plan.get("target_intent", "")).strip(),
        "plant_prompt": plant_prompt,
        "classes": class_reports,
        "fused": fused,
    }
    if provenance is not None:
        report["provenance"] = dict(provenance)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
