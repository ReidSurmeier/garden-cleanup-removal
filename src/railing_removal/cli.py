from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from railing_removal.completion import complete_railing_lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove line-family railings from a plant-candidate PLY using "
            "fused multi-view railing evidence."
        )
    )
    parser.add_argument("--source-cloud", required=True, type=Path)
    parser.add_argument("--candidate-plant", required=True, type=Path)
    parser.add_argument("--railing-votes", required=True, type=Path)
    parser.add_argument("--plant-votes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _vertices(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    return np.asarray(ply["vertex"].data)


def _write_vertices(path: Path, vertices: np.ndarray) -> None:
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,
        byte_order="<",
    ).write(path)


def main() -> None:
    args = _parser().parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    source = _vertices(args.source_cloud.resolve())
    candidate = _vertices(args.candidate_plant.resolve())
    required = {"x", "y", "z", "red", "green", "blue", "source_index"}
    fields = set(source.dtype.names or ())
    if not required <= fields:
        raise ValueError(f"source PLY is missing fields: {sorted(required - fields)}")
    if "source_index" not in (candidate.dtype.names or ()):
        raise ValueError("candidate PLY is missing source_index")
    source_ids = np.asarray(source["source_index"])
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("source_index values must be unique")
    candidate_mask = np.isin(source_ids, candidate["source_index"])
    if int(candidate_mask.sum()) != len(candidate):
        raise ValueError("candidate PLY contains IDs absent from the source cloud")

    coordinates = np.column_stack((source["x"], source["y"], source["z"]))
    rgb = np.column_stack((source["red"], source["green"], source["blue"]))
    rejected, report = complete_railing_lines(
        coordinates,
        rgb=rgb,
        candidate_mask=candidate_mask,
        railing_votes=np.load(args.railing_votes.resolve()),
        plant_votes=np.load(args.plant_votes.resolve()),
    )
    plant_mask = candidate_mask & ~rejected
    report.update(
        {
            "source_cloud": str(args.source_cloud.resolve()),
            "candidate_plant": str(args.candidate_plant.resolve()),
            "plant_before": int(candidate_mask.sum()),
            "plant_after": int(plant_mask.sum()),
            "removed_from_plant": int((candidate_mask & rejected).sum()),
        }
    )

    output.mkdir(parents=True)
    _write_vertices(output / "plant-railing-corrected.ply", source[plant_mask])
    _write_vertices(output / "rejected-railing-corrected.ply", source[~plant_mask])
    (output / "railing-completion-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
