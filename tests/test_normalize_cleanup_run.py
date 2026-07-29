from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from plant_cleanup.plyio import VERTEX_DTYPE
from railing_removal.normalize_run import normalize_cleanup_run


def _write_cloud(path: Path, points: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar classification\n"
        "property uint source_index\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + points.tobytes())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cleanup_run_normalization_uses_ground_decisions_and_preserves_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ply"
    cleanup = tmp_path / "cleanup"
    final = cleanup / "final"
    semantic = cleanup / "semantic"
    points = np.zeros(130, dtype=VERTEX_DTYPE)
    points["x"] = np.linspace(-3.0, 3.0, len(points))
    points["y"] = np.tile(np.linspace(-2.0, 2.0, 13), 10)
    points["z"] = 0.4 * points["y"] - 2.0
    points["z"][100:] += 2.0
    points["nz"] = 1.0
    points["red"] = 25
    points["green"] = 100
    points["blue"] = 40
    points["source_index"] = np.arange(len(points), dtype=np.uint32)
    _write_cloud(source, points)
    _write_cloud(final / "plant-cleaned-color-corrected.ply", points[100:])
    _write_cloud(final / "plant-cleaned-conservative.ply", points[100:])
    _write_cloud(final / "rejected-cleaned.ply", points[:100])
    _write_cloud(semantic / "uncertain-semantic.ply", points[0:0])
    decisions = np.ones(len(points), dtype=np.uint8)
    decisions[:100] = 2
    np.save(final / "decision-codes.npy", decisions)

    normal = np.array((0.0, -0.4, 1.0), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    ground = np.column_stack(
        (points["x"][:100], points["y"][:100], points["z"][:100])
    )
    camera_centers = ground[::25] + 3.0 * normal
    inventory = tmp_path / "camera-inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "project_opened_read_only": True,
                "coordinate_frame": {"source": "mean_aligned_camera_axes"},
                "cameras": [
                    {
                        "enabled": True,
                        "aligned": True,
                        "source_frame_center": center.tolist(),
                        "source_frame_up": [0.0, 0.0, 1.0],
                    }
                    for center in camera_centers
                ],
            }
        ),
        encoding="utf-8",
    )
    source_hash = _sha256(source)

    report = normalize_cleanup_run(
        source,
        cleanup,
        inventory,
        tmp_path / "normalized",
    )

    assert report["ground_evidence"] == {
        "decision_code": 2,
        "point_count": 100,
    }
    assert report["plan"]["status"] == "automatic"
    assert set(report["layers"]) == {
        "source",
        "plant",
        "conservative",
        "rejected",
        "uncertain",
    }
    assert report["cleanup_run"] == str(cleanup.resolve())
    assert report["source_sha256"] == source_hash
    assert _sha256(source) == source_hash
    assert Path(report["report"]).is_file()
