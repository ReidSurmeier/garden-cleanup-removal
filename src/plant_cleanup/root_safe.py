from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from plant_cleanup.classification import ClassificationParameters, Reason
from plant_cleanup.color_correct import ColorParameters
from plant_cleanup.semantic_refine import SemanticParameters


def geometry_decisions(reasons: np.ndarray) -> np.ndarray:
    """Map geometry reasons without discarding the uncertainty boundary."""
    reasons = np.asarray(reasons)
    decisions = np.full(len(reasons), 3, dtype=np.uint8)
    decisions[reasons == Reason.SUPPORT_OR_GROUND] = 2
    decisions[reasons == Reason.UNCERTAIN_NEIGHBOR] = 5
    decisions[np.isin(reasons, [Reason.PLANT_SEED, Reason.PLANT_CONNECTED])] = 1
    return decisions


def validate_config(config: dict) -> None:
    required = {
        "method",
        "target_classification",
        "support_estimation",
        "vision",
        "semantic_refinement",
        "color_correction",
        "proof_render_size",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"root-safe config is missing sections: {missing}")
    supported_versions = {
        "root-safe-v1",
        "root-safe-v2-adaptive",
        "root-safe-v3-evidence-fusion",
        "root-safe-v4-boundary-aware-fusion",
        "scene-aware-v5-object-evidence",
    }
    if config["method"].get("version") not in supported_versions:
        raise ValueError(f"runner requires one of {sorted(supported_versions)}")
    orbit = config["vision"].get("orbit_degrees", [])
    if not orbit or len(orbit) != len(set(orbit)):
        raise ValueError("orbit views must be a nonempty unique list")
    prompt_grid = config["vision"].get("prompt_grid", [])
    if len(prompt_grid) != 2 or any(value < 1 for value in prompt_grid):
        raise ValueError("prompt_grid must contain two positive dimensions")
    ClassificationParameters(**config["target_classification"])
    SemanticParameters(**config["semantic_refinement"])
    ColorParameters(**config["color_correction"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
