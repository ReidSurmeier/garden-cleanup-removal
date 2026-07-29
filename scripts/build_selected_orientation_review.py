from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from plant_cleanup.cloud_render import render_cloud_views  # noqa: E402
from railing_removal.normalization import (  # noqa: E402
    _rotation_between,
    write_normalized_cloud,
)
from railing_removal.orientation_selection import (  # noqa: E402
    select_orientation,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def _matrix(up: list[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rotation_between(
        np.asarray(up, dtype=np.float64),
        np.array((0.0, 0.0, 1.0), dtype=np.float64),
    )
    return matrix


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: build_selected_orientation_review.py "
            "PROJECTS.json FUSED_EVIDENCE_ROOT VISUAL_SELECTIONS.json "
            "OUTPUT_ROOT"
        )
    manifest_path = Path(sys.argv[1]).resolve()
    fused_root = Path(sys.argv[2]).resolve()
    visual_path = Path(sys.argv[3]).resolve()
    output_root = Path(sys.argv[4]).resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    manifest = _read_json(manifest_path)
    visual = _read_json(visual_path)
    if visual.get("schema_version") != 1:
        raise ValueError("visual selections schema_version must be 1")
    selections = visual.get("selections")
    if not isinstance(selections, dict):
        raise ValueError("visual selections must contain selections")
    output_root.mkdir(parents=True)
    batch: list[dict[str, Any]] = []
    for index, item in enumerate(manifest["projects"], start=1):
        scan_id = str(item["scan_id"])
        print(
            f"[{index}/{len(manifest['projects'])}] {scan_id}",
            flush=True,
        )
        fused = _read_json(
            fused_root / scan_id / "fused-orientation.json"
        )
        selection = select_orientation(
            fused,
            visual_selection=selections.get(scan_id),
        )
        scan_dir = Path(item["project"]).resolve().parent
        cleaned = (
            scan_dir / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
        )
        review_dir = output_root / scan_id
        normalized = review_dir / "selected" / "plant-candidate.ply"
        transform = _matrix(selection["up"])
        before = render_cloud_views(
            cleaned,
            review_dir / "before",
            size=1800,
            point_radius=1,
        )
        layer = write_normalized_cloud(
            cleaned,
            normalized,
            transform,
        )
        after = render_cloud_views(
            normalized,
            review_dir / "selected" / "views",
            size=1800,
            point_radius=1,
        )
        orbit = render_cloud_views(
            normalized,
            review_dir / "selected" / "orbit",
            size=1400,
            point_radius=1,
            yaw_degrees=(0, 45, 90, 135, 180, 225, 270, 315),
        )
        report = {
            "schema_version": 1,
            "scan_id": scan_id,
            "source_cleaned_ply": str(cleaned),
            "source_cleaned_ply_opened_read_only": True,
            "selection": selection,
            "matrix": transform.tolist(),
            "candidate_layer": layer,
            "before": before,
            "after": after,
            "orbit": orbit,
        }
        _write_json(review_dir / "review-report.json", report)
        batch.append(
            {
                "scan_id": scan_id,
                "status": "selected",
                "selection_basis": selection["selection_basis"],
                "report": str(review_dir / "review-report.json"),
            }
        )
    _write_json(
        output_root / "batch-report.json",
        {
            "schema_version": 1,
            "source_cleaned_ply_opened_read_only": True,
            "results": batch,
        },
    )
    print(json.dumps(batch, indent=2), flush=True)


if __name__ == "__main__":
    main()
