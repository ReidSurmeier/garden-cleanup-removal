from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from railing_removal.metashape_reader import export_reader_cloud_readonly


def _fake_metashape() -> object:
    points = [
        SimpleNamespace(
            position=(float(index), float(index + 1), float(index + 2)),
            normal=(0.0, 0.0, 1.0),
            color=(10 + index, 20 + index, 30 + index),
            classification=index,
        )
        for index in range(3)
    ]

    class FakeReader:
        def __init__(self) -> None:
            self.position = 0

        def open(self, cloud: object) -> None:
            self.position = 0

        def read(self, count: int) -> list[object]:
            result = points[self.position : self.position + count]
            self.position += len(result)
            return result

    class FakeDocument:
        read_only = False
        chunks = [
            SimpleNamespace(
                point_cloud=SimpleNamespace(point_count=3),
                cameras=[
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
                ],
            )
        ]

        def open(self, path: str, read_only: bool) -> None:
            self.read_only = read_only

    return SimpleNamespace(
        version="2.3.1-test",
        Document=FakeDocument,
        PointCloud=SimpleNamespace(Reader=FakeReader),
    )


def test_export_preserves_source_indices_colors_and_read_only_mode(
    tmp_path: Path,
) -> None:
    project = tmp_path / "plant.psx"
    project.write_text("project", encoding="utf-8")
    output = tmp_path / "cloud.ply"

    report = export_reader_cloud_readonly(
        project,
        output,
        _fake_metashape(),
        stride=2,
        chunk_size=2,
    )

    payload = output.read_bytes()
    header, body = payload.split(b"end_header\n", maxsplit=1)
    assert b"element vertex 2" in header
    record = struct.Struct("<ffffffBBBBI")
    first = record.unpack_from(body, 0)
    second = record.unpack_from(body, record.size)
    assert first[:3] == (0.0, -2.0, 1.0)
    assert first[3:6] == (0.0, -1.0, 0.0)
    assert first[6:10] == (10, 20, 30, 0)
    assert first[10] == 0
    assert second[6:10] == (12, 22, 32, 2)
    assert second[10] == 2
    assert report["read_only"] is True
    assert report["exported_point_count"] == 2


def test_export_refuses_to_overwrite_an_existing_file(tmp_path: Path) -> None:
    project = tmp_path / "plant.psx"
    project.write_text("project", encoding="utf-8")
    output = tmp_path / "cloud.ply"
    output.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        export_reader_cloud_readonly(
            project,
            output,
            _fake_metashape(),
        )

    assert output.read_bytes() == b"keep me"
