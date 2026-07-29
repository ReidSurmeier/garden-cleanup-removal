from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from railing_removal.metashape_inventory import build_camera_inventory


def test_inventory_records_camera_and_metric_evidence_in_source_frame() -> None:
    identity = np.eye(4, dtype=np.float64).ravel().tolist()
    camera = SimpleNamespace(
        label="00001.png",
        enabled=True,
        transform=identity,
        center=(1.0, 2.0, 3.0),
        photo=SimpleNamespace(path=r"F:\3d_scans\frames\00001.png"),
        sensor=None,
        reference=SimpleNamespace(
            enabled=True,
            location=(37.0, -122.0, 14.0),
            location_accuracy=(5.0, 5.0, 8.0),
        ),
    )
    scalebar = SimpleNamespace(
        label="known-rail-width",
        reference=SimpleNamespace(
            enabled=True,
            distance=1.25,
            accuracy=0.005,
        ),
    )
    chunk = SimpleNamespace(
        label="Chunk 1",
        cameras=[camera],
        point_cloud=SimpleNamespace(point_count=1000),
        depth_maps={},
        transform=SimpleNamespace(scale=0.25, matrix=identity),
        crs=SimpleNamespace(name="Local Coordinates"),
        scalebars=[scalebar],
    )

    report = build_camera_inventory(
        chunk,
        project=Path(r"F:\3d_scans\scan\capture.psx"),
        metashape_version="2.3.1",
        project_opened_read_only=True,
    )

    assert report["project_opened_read_only"] is True
    assert report["coordinate_frame"]["source"] == "mean_aligned_camera_axes"
    assert report["chunk_transform"]["scale"] == 0.25
    assert report["coordinate_reference"]["name"] == "Local Coordinates"
    assert report["scale_bars"] == [
        {
            "accuracy": 0.005,
            "distance": 1.25,
            "enabled": True,
            "label": "known-rail-width",
        }
    ]
    assert report["reference_summary"] == {
        "enabled_camera_locations": 1,
        "enabled_scale_bars": 1,
    }
    np.testing.assert_allclose(
        report["cameras"][0]["source_frame_center"],
        (1.0, 3.0, -2.0),
    )
    np.testing.assert_allclose(
        report["cameras"][0]["source_frame_up"],
        (0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(
        report["cameras"][0]["source_frame_right"],
        (1.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        report["cameras"][0]["source_frame_down"],
        (0.0, 0.0, -1.0),
    )
    np.testing.assert_allclose(
        report["cameras"][0]["source_frame_forward"],
        (0.0, 1.0, 0.0),
    )
