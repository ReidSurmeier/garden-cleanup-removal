from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import Metashape

from railing_removal.camera_inventory_batch import (
    run_camera_inventory_batch,
)


def _vector(value: object | None) -> list[float] | None:
    return [float(item) for item in value] if value is not None else None


def _inventory_project(project: Path) -> dict[str, Any]:
    document = Metashape.Document()
    try:
        document.open(str(project), read_only=True)
        if not bool(document.read_only):
            raise RuntimeError("Metashape did not open the project read-only")
        chunk = document.chunks[0]
        cameras = []
        for camera in chunk.cameras:
            calibration = (
                camera.sensor.calibration
                if camera.sensor is not None
                else None
            )
            cameras.append(
                {
                    "label": str(camera.label),
                    "enabled": bool(camera.enabled),
                    "aligned": camera.transform is not None,
                    "photo": (
                        str(camera.photo.path)
                        if camera.photo is not None
                        else None
                    ),
                    "center": _vector(camera.center),
                    "transform": _vector(camera.transform),
                    "calibration": (
                        {
                            "width": int(calibration.width),
                            "height": int(calibration.height),
                            "f": float(calibration.f),
                            "cx": float(calibration.cx),
                            "cy": float(calibration.cy),
                            "b1": float(calibration.b1),
                            "b2": float(calibration.b2),
                            "k1": float(calibration.k1),
                            "k2": float(calibration.k2),
                            "k3": float(calibration.k3),
                            "k4": float(calibration.k4),
                            "p1": float(calibration.p1),
                            "p2": float(calibration.p2),
                        }
                        if calibration is not None
                        else None
                    ),
                }
            )
        return {
            "schema_version": 1,
            "metashape_version": str(Metashape.version),
            "project": str(project),
            "project_opened_read_only": True,
            "chunk": str(chunk.label),
            "point_count": (
                int(chunk.point_cloud.point_count)
                if chunk.point_cloud is not None
                else 0
            ),
            "camera_count": len(cameras),
            "aligned_camera_count": sum(
                camera["aligned"] for camera in cameras
            ),
            "depth_map_count": len(chunk.depth_maps.keys()),
            "cameras": cameras,
        }
    finally:
        del document
        gc.collect()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: metashape_camera_inventory_batch.py "
            "PROJECTS.json OUTPUT_ROOT"
        )
    report = run_camera_inventory_batch(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        inventory_project=_inventory_project,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
