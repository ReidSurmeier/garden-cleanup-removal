from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.orientation_consensus import (  # noqa: E402
    OrientationEvidence,
    resolve_orientation_consensus,
)
from railing_removal.photo_orientation import (  # noqa: E402
    CameraProjection,
    LineSegment,
    detect_vertical_line_segments,
    estimate_vertical_from_segments,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def _camera_projection(camera: dict[str, Any]) -> CameraProjection:
    calibration = camera["calibration"]
    return CameraProjection(
        right=np.asarray(camera["source_frame_right"], dtype=np.float64),
        down=np.asarray(camera["source_frame_down"], dtype=np.float64),
        forward=np.asarray(camera["source_frame_forward"], dtype=np.float64),
        focal_length=float(calibration["f"]),
        principal_x=float(calibration["width"]) / 2.0
        + float(calibration["cx"]),
        principal_y=float(calibration["height"]) / 2.0
        + float(calibration["cy"]),
    )


def _evenly_sample(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[index] for index in indices]


def _photo_path(
    camera: dict[str, Any],
    *,
    scan_id: str,
    photo_root: Path,
) -> Path:
    recorded = Path(str(camera["photo"]))
    if recorded.is_file():
        return recorded
    return photo_root / scan_id / "png" / recorded.name


def _draw_overlay(
    image: np.ndarray,
    segments: list[LineSegment],
    output: Path,
) -> None:
    preview = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 2
        else image.copy()
    )
    for segment in segments:
        cv2.line(
            preview,
            (round(segment.x1), round(segment.y1)),
            (round(segment.x2), round(segment.y2)),
            (0, 255, 255),
            max(2, round(max(preview.shape[:2]) / 800)),
            cv2.LINE_AA,
        )
    scale = min(1.0, 1600.0 / max(preview.shape[:2]))
    if scale < 1.0:
        preview = cv2.resize(
            preview,
            (
                round(preview.shape[1] * scale),
                round(preview.shape[0] * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output),
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    ):
        raise OSError(f"failed to write line overlay: {output}")


def _ground_evidence(report: dict[str, Any]) -> list[OrientationEvidence]:
    floor = report.get("floor")
    components = (
        floor.get("floor_components", [])
        if isinstance(floor, dict)
        else []
    )
    result: list[OrientationEvidence] = []
    for component in components:
        normal = component.get("normal")
        points = max(
            int(component.get("matched_source_point_count", 0)),
            int(component.get("seed_point_count", 0)),
        )
        if normal is None or points < 500:
            continue
        vector = np.asarray(normal, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            continue
        if vector[2] < 0:
            vector = -vector
        result.append(
            OrientationEvidence(
                family="ground",
                source=f"connected-floor-component-{component['label']}",
                up=vector,
                confidence=min(1.0, 0.55 + np.log10(points) / 12.0),
                metadata={"point_count": points},
            )
        )
    return result


def _camera_evidence(inventory: dict[str, Any]) -> OrientationEvidence:
    vectors = [
        np.asarray(camera["source_frame_up"], dtype=np.float64)
        for camera in inventory["cameras"]
        if camera.get("enabled")
        and camera.get("aligned")
        and camera.get("source_frame_up") is not None
    ]
    consensus = np.sum(
        [vector / np.linalg.norm(vector) for vector in vectors],
        axis=0,
    )
    return OrientationEvidence(
        family="camera",
        source="aligned-camera-image-up-consensus",
        up=consensus,
        confidence=0.65,
        metadata={"camera_count": len(vectors)},
    )


def analyze_scan(
    *,
    scan_id: str,
    inventory_path: Path,
    report_path: Path,
    photo_root: Path,
    output: Path,
    maximum_photos: int = 16,
) -> dict[str, Any]:
    inventory = _read_json(inventory_path)
    report = _read_json(report_path)
    usable_cameras = [
        camera
        for camera in inventory["cameras"]
        if camera.get("enabled")
        and camera.get("aligned")
        and camera.get("calibration")
        and camera.get("source_frame_right") is not None
        and camera.get("source_frame_down") is not None
        and camera.get("source_frame_forward") is not None
        and camera.get("photo")
    ]
    selected = _evenly_sample(usable_cameras, maximum_photos)
    observations: list[tuple[CameraProjection, LineSegment]] = []
    photo_records: list[dict[str, Any]] = []
    cameras_with_lines = 0
    for camera in selected:
        photo = _photo_path(
            camera,
            scan_id=scan_id,
            photo_root=photo_root,
        )
        image = cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE)
        if image is None:
            photo_records.append(
                {
                    "camera": camera["label"],
                    "photo": str(photo),
                    "status": "missing",
                }
            )
            continue
        segments = detect_vertical_line_segments(image)
        if segments:
            cameras_with_lines += 1
            projection = _camera_projection(camera)
            observations.extend((projection, segment) for segment in segments)
            _draw_overlay(
                image,
                segments,
                output / "line-overlays" / f"{camera['label']}.jpg",
            )
        photo_records.append(
            {
                "camera": camera["label"],
                "photo": str(photo),
                "status": "usable",
                "vertical_line_count": len(segments),
            }
        )

    camera_item = _camera_evidence(inventory)
    photo_result = estimate_vertical_from_segments(
        observations,
        reference_up=camera_item.up,
        minimum_inlier_segments=max(8, cameras_with_lines),
    )
    evidence = [camera_item, *_ground_evidence(report)]
    if photo_result["status"] == "usable" and cameras_with_lines >= 3:
        evidence.append(
            OrientationEvidence(
                family="photo",
                source="multi-view-long-line-constraints",
                up=np.asarray(photo_result["up"], dtype=np.float64),
                confidence=min(
                    0.95,
                    0.55
                    + 0.4
                    * photo_result["inlier_segment_count"]
                    / max(1, photo_result["segment_count"]),
                ),
                metadata={
                    "cameras_with_lines": cameras_with_lines,
                    "inlier_segment_count": photo_result[
                        "inlier_segment_count"
                    ],
                },
            )
        )
    consensus = resolve_orientation_consensus(evidence)
    result = {
        "schema_version": 1,
        "scan_id": scan_id,
        "source_projects_opened_read_only": True,
        "source_ply_opened_read_only": True,
        "inventory": str(inventory_path),
        "cleanup_report": str(report_path),
        "selected_photo_count": len(selected),
        "cameras_with_lines": cameras_with_lines,
        "photo_orientation": photo_result,
        "photos": photo_records,
        "evidence": [
            {
                "family": item.family,
                "source": item.source,
                "up": item.up.tolist(),
                "confidence": item.confidence,
                "metadata": item.metadata,
            }
            for item in evidence
        ],
        "consensus": consensus,
    }
    _write_json(output / "orientation-evidence.json", result)
    return result


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: analyze_photo_orientation_batch.py "
            "PROJECTS.json INVENTORY_ROOT PHOTO_ROOT OUTPUT_ROOT"
        )
    manifest_path = Path(sys.argv[1]).resolve()
    inventory_root = Path(sys.argv[2]).resolve()
    photo_root = Path(sys.argv[3]).resolve()
    output_root = Path(sys.argv[4]).resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    manifest = _read_json(manifest_path)
    output_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for index, item in enumerate(manifest["projects"], start=1):
        scan_id = str(item["scan_id"])
        scan_dir = Path(item["project"]).resolve().parent
        reports = sorted(
            scan_dir.glob("plant-cleanup-report-garden-*-final-v2.json")
        )
        if len(reports) != 1:
            raise ValueError(
                f"{scan_id} requires exactly one final-v2 cleanup report"
            )
        print(
            f"[{index}/{len(manifest['projects'])}] {scan_id}",
            flush=True,
        )
        result = analyze_scan(
            scan_id=scan_id,
            inventory_path=(
                inventory_root / scan_id / "camera-inventory.json"
            ),
            report_path=reports[0],
            photo_root=photo_root,
            output=output_root / scan_id,
        )
        results.append(
            {
                "scan_id": scan_id,
                "status": result["consensus"]["status"],
                "consensus_up": result["consensus"]["consensus_up"],
                "supporting_families": result["consensus"][
                    "supporting_families"
                ],
                "cameras_with_lines": result["cameras_with_lines"],
                "photo_inliers": result["photo_orientation"].get(
                    "inlier_segment_count",
                    0,
                ),
            }
        )
    _write_json(
        output_root / "batch-report.json",
        {
            "schema_version": 1,
            "manifest": str(manifest_path),
            "source_data_opened_read_only": True,
            "results": results,
        },
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
