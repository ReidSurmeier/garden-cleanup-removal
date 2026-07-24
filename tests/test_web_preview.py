from __future__ import annotations

from pathlib import Path

import numpy as np

from plant_cleanup.plyio import VERTEX_DTYPE
from plant_cleanup.web_preview import export_web_preview
from railing_removal.full_pipeline import _build_viewer


def _write_cloud(path: Path, points: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar classification\n"
        "property uint source_index\n"
        "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + points.tobytes())


def test_empty_preview_is_valid_and_does_not_distort_viewer_bounds(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.ply"
    populated = tmp_path / "populated.ply"
    _write_cloud(empty, np.zeros(0, dtype=VERTEX_DTYPE))
    points = np.zeros(2, dtype=VERTEX_DTYPE)
    points["x"] = (100.0, 104.0)
    points["y"] = (200.0, 208.0)
    points["z"] = (300.0, 312.0)
    points["source_index"] = (0, 1)
    _write_cloud(populated, points)

    empty_report = export_web_preview(
        empty,
        tmp_path / "empty.bin",
    )
    manifest = _build_viewer(
        source=populated,
        previous=populated,
        plant=populated,
        conservative=populated,
        rejected=populated,
        uncertain=empty,
        output=tmp_path / "review",
    )

    assert empty_report["preview_point_count"] == 0
    assert empty_report["bounds"] is None
    assert manifest["bounds"] == {
        "min": [100.0, 200.0, 300.0],
        "max": [104.0, 208.0, 312.0],
    }
