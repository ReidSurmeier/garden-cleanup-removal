from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from plant_cleanup.plyio import read_cloud
from railing_removal.normalization import (
    NormalizationParameters,
    normalize_cleanup_layers,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_cleanup_run(
    source_path: Path,
    cleanup_run: Path,
    camera_inventory_path: Path,
    output_dir: Path,
    *,
    parameters: NormalizationParameters = NormalizationParameters(),
) -> dict[str, Any]:
    """Normalize a frozen cleanup run without changing its source artifacts."""

    source_path = source_path.resolve()
    cleanup_run = cleanup_run.resolve()
    camera_inventory_path = camera_inventory_path.resolve()
    output_dir = output_dir.resolve()
    final_dir = cleanup_run / "final"
    decision_path = final_dir / "decision-codes.npy"
    layers = {
        "plant": final_dir / "plant-cleaned-color-corrected.ply",
        "conservative": final_dir / "plant-cleaned-conservative.ply",
        "rejected": final_dir / "rejected-cleaned.ply",
        "uncertain": cleanup_run / "semantic" / "uncertain-semantic.ply",
    }
    required = [
        source_path,
        camera_inventory_path,
        decision_path,
        *layers.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cleanup normalization inputs are missing: " + ", ".join(missing)
        )

    source = read_cloud(source_path)
    decisions = np.load(decision_path, mmap_mode="r")
    if decisions.shape != (len(source),):
        raise ValueError(
            "cleanup decision codes must match the source point count"
        )
    ground_mask = np.asarray(decisions == 2, dtype=bool)
    inventory = json.loads(
        camera_inventory_path.read_text(encoding="utf-8")
    )
    normalization = normalize_cleanup_layers(
        source_path,
        layers,
        ground_mask=ground_mask,
        camera_inventory=inventory,
        output_dir=output_dir,
        parameters=parameters,
    )
    report = {
        **normalization,
        "cleanup_run": str(cleanup_run),
        "cleanup_decisions": str(decision_path),
        "cleanup_decisions_sha256": _sha256(decision_path),
        "camera_inventory_path": str(camera_inventory_path),
        "camera_inventory_sha256": _sha256(camera_inventory_path),
        "ground_evidence": {
            "decision_code": 2,
            "point_count": int(ground_mask.sum()),
        },
    }
    report_path = output_dir / "cleanup-normalization-report.json"
    with report_path.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return {**report, "report": str(report_path)}
