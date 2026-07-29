from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from plant_cleanup.plyio import read_cloud  # noqa: E402
from railing_removal.normalization import (  # noqa: E402
    camera_evidence_from_inventory,
    estimate_normalization_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score cleanup-ground and Metashape camera orientation evidence "
            "without writing point clouds."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    for item in manifest["scans"]:
        source = read_cloud(Path(item["source"]))
        decisions = np.load(item["decision_codes"], mmap_mode="r")
        inventory = json.loads(
            Path(item["camera_inventory"]).read_text(encoding="utf-8")
        )
        semantic_report = json.loads(
            Path(item["semantic_report"]).read_text(encoding="utf-8")
        )
        support_plane = semantic_report.get("support_plane")
        if not isinstance(support_plane, dict):
            raise ValueError(
                f"{item['scan_id']} lacks measured support-plane evidence"
            )
        centers, up_vectors, _ = camera_evidence_from_inventory(inventory)
        coordinates = np.column_stack(
            (source["x"], source["y"], source["z"])
        )
        try:
            plan = estimate_normalization_plan(
                coordinates,
                ground_mask=decisions == 2,
                camera_centers=centers,
                camera_up_vectors=up_vectors,
                support_plane=support_plane,
            )
            results.append(
                {
                    "scan_id": item["scan_id"],
                    "status": plan["status"],
                    "ground_points": plan["evidence"]["ground"][
                        "candidate_point_count"
                    ],
                    "ground_planarity": plan["evidence"]["ground"][
                        "planarity"
                    ],
                    "ground_normal": plan["evidence"]["ground"]["normal"],
                    "orientation_basis": plan["evidence"][
                        "orientation_basis"
                    ],
                    "camera_count": plan["evidence"]["cameras"][
                        "aligned_camera_count"
                    ],
                    "view_up_disagreement_degrees": plan["evidence"][
                        "cameras"
                    ]["up_disagreement_degrees"],
                    "scale": plan["scale"],
                    "matrix": plan["matrix"],
                }
            )
        except Exception as error:
            results.append(
                {
                    "scan_id": item["scan_id"],
                    "status": "failed",
                    "error": str(error),
                }
            )
        del coordinates, decisions, source
        gc.collect()
    report = {"schema_version": 1, "scans": results}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(
                f"refusing to overwrite orientation report: {args.output}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
