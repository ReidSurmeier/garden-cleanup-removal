from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from plant_cleanup.plyio import read_cloud  # noqa: E402
from railing_removal.geometry_orientation import (  # noqa: E402
    estimate_rigid_axes_from_cloud,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def _cleanup_report(scan_dir: Path) -> Path:
    reports = sorted(
        scan_dir.glob("plant-cleanup-report-garden-*-final-v2.json")
    )
    if len(reports) != 1:
        raise ValueError("scan requires exactly one final-v2 cleanup report")
    return reports[0]


def analyze_scan(
    scan_id: str,
    report_path: Path,
    *,
    maximum_input_points: int = 300_000,
) -> dict[str, Any]:
    cleanup = _read_json(report_path)
    source_path = Path(cleanup["source"]).resolve()
    cloud = read_cloud(source_path)
    if len(cloud) > maximum_input_points:
        indices = np.linspace(
            0,
            len(cloud) - 1,
            maximum_input_points,
        ).round().astype(int)
        cloud = cloud[indices]
    coordinates = np.column_stack(
        (cloud["x"], cloud["y"], cloud["z"])
    )
    normals = np.column_stack(
        (cloud["nx"], cloud["ny"], cloud["nz"])
    )
    colors = np.column_stack(
        (cloud["red"], cloud["green"], cloud["blue"])
    )
    geometry = estimate_rigid_axes_from_cloud(
        coordinates,
        normals,
        colors,
        reference_up=np.array((0.0, 0.0, 1.0)),
    )
    return {
        "schema_version": 1,
        "scan_id": scan_id,
        "source": str(source_path),
        "source_opened_read_only": True,
        "source_point_count": int(cleanup["source_point_count"]),
        "analysis_input_point_count": int(len(cloud)),
        "geometry": geometry,
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: analyze_geometry_orientation_batch.py "
            "PROJECTS.json OUTPUT_ROOT"
        )
    manifest_path = Path(sys.argv[1]).resolve()
    output_root = Path(sys.argv[2]).resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    manifest = _read_json(manifest_path)
    output_root.mkdir(parents=True)
    summary: list[dict[str, Any]] = []
    for index, item in enumerate(manifest["projects"], start=1):
        scan_id = str(item["scan_id"])
        print(
            f"[{index}/{len(manifest['projects'])}] {scan_id}",
            flush=True,
        )
        report = analyze_scan(
            scan_id,
            _cleanup_report(Path(item["project"]).resolve().parent),
        )
        destination = output_root / scan_id / "geometry-orientation.json"
        _write_json(destination, report)
        summary.append(
            {
                "scan_id": scan_id,
                "status": report["geometry"]["status"],
                "candidates": report["geometry"]["candidates"],
            }
        )
    _write_json(
        output_root / "batch-report.json",
        {
            "schema_version": 1,
            "manifest": str(manifest_path),
            "source_ply_opened_read_only": True,
            "results": summary,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
