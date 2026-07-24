from __future__ import annotations

import json
import sys
from pathlib import Path

import Metashape


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: metashape_projection_sample.py PROJECT.psx OUTPUT.json"
        )
    project = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sample: {output}")
    document = Metashape.Document()
    document.open(str(project), read_only=True)
    if not bool(document.read_only):
        raise RuntimeError("Metashape did not open the project read-only")
    chunk = document.chunks[0]
    cloud = chunk.point_cloud
    if cloud is None:
        raise ValueError("project contains no point cloud")
    cameras = [
        camera
        for camera in chunk.cameras
        if camera.enabled and camera.transform is not None
    ]
    selected_cameras = [
        cameras[round(index * (len(cameras) - 1) / 4)]
        for index in range(5)
    ]
    point_count = int(cloud.point_count)
    step = max(point_count // 1_000, 1)
    reader = Metashape.PointCloud.Reader()
    reader.open(cloud)
    points = []
    source_index = 0
    while True:
        block = reader.read(100_000)
        if not block:
            break
        for point in block:
            if source_index % step == 0:
                projections = {}
                for camera in selected_cameras:
                    projected = camera.project(point.position)
                    camera_point = camera.transform.inv().mulp(
                        point.position
                    )
                    projections[str(camera.label)] = {
                        "pixel": (
                            [float(projected[0]), float(projected[1])]
                            if projected is not None
                            else None
                        ),
                        "camera_coordinates": [
                            float(camera_point[index])
                            for index in range(3)
                        ],
                    }
                points.append(
                    {
                        "source_index": source_index,
                        "position": [
                            float(point.position[index])
                            for index in range(3)
                        ],
                        "projections": projections,
                    }
                )
            source_index += 1
    if source_index != point_count:
        raise RuntimeError(
            f"reader count mismatch: {source_index} != {point_count}"
        )
    report = {
        "schema_version": 1,
        "metashape_version": str(Metashape.version),
        "project": str(project),
        "project_opened_read_only": True,
        "point_count": point_count,
        "sample_step": step,
        "sample_count": len(points),
        "camera_labels": [
            str(camera.label) for camera in selected_cameras
        ],
        "points": points,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(
        json.dumps(
            {
                "project_opened_read_only": True,
                "point_count": point_count,
                "sample_count": len(points),
                "camera_labels": report["camera_labels"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
