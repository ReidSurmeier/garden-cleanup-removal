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
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: build_fused_orientation_review.py "
            "PROJECTS.json FUSED_EVIDENCE_ROOT OUTPUT_ROOT"
        )
    manifest_path = Path(sys.argv[1]).resolve()
    fused_root = Path(sys.argv[2]).resolve()
    output_root = Path(sys.argv[3]).resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    manifest = _read_json(manifest_path)
    output_root.mkdir(parents=True)
    batch: list[dict[str, Any]] = []
    for index, item in enumerate(manifest["projects"], start=1):
        scan_id = str(item["scan_id"])
        print(
            f"[{index}/{len(manifest['projects'])}] {scan_id}",
            flush=True,
        )
        scan_dir = Path(item["project"]).resolve().parent
        cleaned = (
            scan_dir / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
        )
        fused = _read_json(
            fused_root / scan_id / "fused-orientation.json"
        )
        review_dir = output_root / scan_id
        before = render_cloud_views(
            cleaned,
            review_dir / "before",
            size=1800,
            point_radius=1,
        )
        candidates = fused["consensus"]["ranked_candidates"]
        candidate_limit = (
            1 if fused["consensus"]["status"] == "automatic" else 3
        )
        if fused["consensus"]["status"] == "automatic":
            candidates = [
                {
                    **candidates[0],
                    "cluster_consensus_up": candidates[0][
                        "consensus_up"
                    ],
                    "consensus_up": fused["consensus"]["selected_up"],
                    "selection_basis": fused["consensus"][
                        "selection_basis"
                    ],
                }
            ]
        candidate_reports: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(
            candidates[:candidate_limit],
            start=1,
        ):
            candidate_dir = review_dir / f"candidate-{candidate_index}"
            normalized = candidate_dir / "plant-candidate.ply"
            transform = _matrix(candidate["consensus_up"])
            layer = write_normalized_cloud(
                cleaned,
                normalized,
                transform,
            )
            renders = render_cloud_views(
                normalized,
                candidate_dir / "views",
                size=1800,
                point_radius=1,
            )
            candidate_reports.append(
                {
                    "candidate": candidate_index,
                    "matrix": transform.tolist(),
                    "hypothesis": candidate,
                    "layer": layer,
                    "renders": renders,
                }
            )
        report = {
            "schema_version": 1,
            "scan_id": scan_id,
            "source_cleaned_ply": str(cleaned),
            "source_cleaned_ply_opened_read_only": True,
            "decision_status": fused["consensus"]["status"],
            "decision_reason": fused["consensus"]["reason"],
            "before": before,
            "candidates": candidate_reports,
        }
        _write_json(review_dir / "review-report.json", report)
        batch.append(
            {
                "scan_id": scan_id,
                "status": report["decision_status"],
                "candidate_count": len(candidate_reports),
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
