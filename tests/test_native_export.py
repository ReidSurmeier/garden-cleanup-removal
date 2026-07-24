from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from plant_cleanup.plyio import read_cloud
from railing_removal.native_export import (
    canonicalize_native_cloud,
    export_native_cloud_readonly,
)


NATIVE_RECORD = struct.Struct("<ffffffBBBB")


def _write_native(path: Path) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar class\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("xb") as destination:
        destination.write(header)
        for index in range(3):
            destination.write(
                NATIVE_RECORD.pack(
                    float(index),
                    float(index + 10),
                    float(index + 20),
                    1.0,
                    0.0,
                    0.0,
                    10 + index,
                    20 + index,
                    30 + index,
                    index,
                )
            )


def _fake_metashape() -> object:
    class FakeChunk:
        point_cloud = SimpleNamespace(point_count=3)
        cameras = [
            SimpleNamespace(
                transform=(
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                )
            )
        ]

        def exportPointCloud(self, path: str, **kwargs: object) -> None:
            self.export_kwargs = kwargs
            _write_native(Path(path))

    chunk = FakeChunk()

    class FakeDocument:
        read_only = False
        chunks = [chunk]

        def open(self, path: str, read_only: bool) -> None:
            self.read_only = read_only

    return SimpleNamespace(
        version="2.3.0-test",
        Document=FakeDocument,
        DataSource=SimpleNamespace(PointCloudData="point-cloud"),
        PointCloudFormatPLY="ply",
        chunk=chunk,
    )


def test_native_export_uses_read_only_project_and_atomic_final_name(
    tmp_path: Path,
) -> None:
    project = tmp_path / "scan.psx"
    project.write_text("source", encoding="utf-8")
    output = tmp_path / "native.ply"
    metashape = _fake_metashape()

    report = export_native_cloud_readonly(project, output, metashape)

    assert output.is_file()
    assert not list(tmp_path.glob("native.ply.partial-*"))
    assert report["read_only"] is True
    assert report["source_point_count"] == 3
    assert metashape.chunk.export_kwargs["source_data"] == "point-cloud"
    assert metashape.chunk.export_kwargs["save_point_classification"] is True


def test_canonicalizer_rotates_and_strides_native_points(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native.ply"
    _write_native(source)
    output = tmp_path / "canonical.ply"
    frame = {
        "right": [0.0, 1.0, 0.0],
        "forward": [1.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "source": "test",
    }

    report = canonicalize_native_cloud(
        source,
        output,
        frame,
        stride=2,
    )

    cloud = read_cloud(output)
    assert len(cloud) == 2
    np.testing.assert_allclose(cloud["x"], [10.0, 12.0])
    np.testing.assert_allclose(cloud["y"], [0.0, 2.0])
    np.testing.assert_allclose(cloud["z"], [20.0, 22.0])
    np.testing.assert_allclose(cloud["nx"], [0.0, 0.0])
    np.testing.assert_allclose(cloud["ny"], [1.0, 1.0])
    assert cloud["source_index"].tolist() == [0, 2]
    assert cloud["classification"].tolist() == [0, 2]
    assert report["exported_point_count"] == 2


def test_canonicalizer_closes_memory_maps_before_atomic_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "native.ply"
    _write_native(source)
    output = tmp_path / "canonical.ply"
    memory_maps: list[np.memmap] = []
    original_memmap = np.memmap
    original_rename = Path.rename

    def tracked_memmap(*args, **kwargs):
        mapped = original_memmap(*args, **kwargs)
        memory_maps.append(mapped)
        return mapped

    def windows_style_rename(path: Path, target: Path) -> Path:
        assert memory_maps
        assert all(mapped._mmap.closed for mapped in memory_maps)
        return original_rename(path, target)

    monkeypatch.setattr(np, "memmap", tracked_memmap)
    monkeypatch.setattr(Path, "rename", windows_style_rename)

    canonicalize_native_cloud(
        source,
        output,
        {
            "right": [1.0, 0.0, 0.0],
            "forward": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "source": "test",
        },
    )

    assert output.is_file()
