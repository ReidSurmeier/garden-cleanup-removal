from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from railing_removal.metashape_reader import _camera_coordinate_frame


NATIVE_PROPERTIES = [
    ("float", "x"),
    ("float", "y"),
    ("float", "z"),
    ("float", "nx"),
    ("float", "ny"),
    ("float", "nz"),
    ("uchar", "red"),
    ("uchar", "green"),
    ("uchar", "blue"),
    ("uchar", "class"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as source:
        lines: list[str] = []
        while True:
            raw = source.readline()
            if not raw:
                raise ValueError(f"missing PLY end_header in {path}")
            line = raw.decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
        offset = source.tell()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("native PLY must be binary little endian")
    count_lines = [
        line for line in lines if line.startswith("element vertex ")
    ]
    if len(count_lines) != 1:
        raise ValueError("native PLY must contain one vertex count")
    properties = [
        tuple(line.split()[1:3])
        for line in lines
        if line.startswith("property ")
    ]
    if properties != NATIVE_PROPERTIES:
        raise ValueError(f"unexpected native PLY properties: {properties}")
    return int(count_lines[0].split()[2]), offset


def export_native_cloud_readonly(
    project: Path,
    output: Path,
    metashape: Any,
) -> dict[str, Any]:
    """Use Metashape's native exporter without saving its source project."""

    project = project.resolve()
    output = output.resolve()
    if not project.is_file():
        raise FileNotFoundError(project)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite native export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    document = metashape.Document()
    document.open(str(project), read_only=True)
    if not bool(document.read_only):
        raise RuntimeError("Metashape did not open the project read-only")
    if not document.chunks:
        raise ValueError("Metashape project contains no chunks")
    chunk = document.chunks[0]
    cloud = chunk.point_cloud
    if cloud is None:
        raise ValueError("first Metashape chunk contains no point cloud")
    frame = _camera_coordinate_frame(chunk)

    partial = output.with_name(f"{output.name}.partial-{uuid4().hex}")
    chunk.exportPointCloud(
        str(partial),
        source_data=metashape.DataSource.PointCloudData,
        format=metashape.PointCloudFormatPLY,
        binary=True,
        save_point_color=True,
        save_point_normal=True,
        save_point_classification=True,
        colors_rgb_8bit=True,
    )
    if not partial.is_file():
        raise RuntimeError("Metashape native export did not create a file")
    count, _ = _native_header(partial)
    expected = int(cloud.point_count)
    if count != expected:
        raise RuntimeError(
            f"native export count mismatch: expected {expected}, got {count}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite native export: {output}")
    partial.rename(output)
    return {
        "metashape_version": str(metashape.version),
        "project": str(project),
        "output": str(output),
        "read_only": True,
        "source_point_count": count,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "coordinate_frame": frame,
    }


def canonicalize_native_cloud(
    source: Path,
    output: Path,
    coordinate_frame: dict[str, Any],
    *,
    stride: int = 1,
    block_size: int = 1_000_000,
) -> dict[str, Any]:
    """Vectorize native PLY into the accepted Z-up/source-index contract."""

    if stride < 1 or block_size < 1:
        raise ValueError("stride and block_size must be positive")
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite canonical export: {output}"
        )

    import numpy as np

    from plant_cleanup.plyio import VERTEX_DTYPE

    count, source_offset = _native_header(source)
    native_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("classification", "u1"),
        ]
    )
    expected_bytes = source_offset + count * native_dtype.itemsize
    if source.stat().st_size != expected_bytes:
        raise ValueError(
            f"native PLY byte count mismatch: expected {expected_bytes}, "
            f"got {source.stat().st_size}"
        )
    frame = np.asarray(
        [
            coordinate_frame["right"],
            coordinate_frame["forward"],
            coordinate_frame["up"],
        ],
        dtype=np.float64,
    ).T
    if frame.shape != (3, 3) or not np.isfinite(frame).all():
        raise ValueError("coordinate frame must be a finite 3x3 basis")

    exported_count = (count + stride - 1) // stride
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment source_index is the sequential Metashape point index\n"
        f"comment coordinate_frame {coordinate_frame.get('source', 'unknown')}\n"
        f"element vertex {exported_count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar classification\n"
        "property uint source_index\n"
        "end_header\n"
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial-{uuid4().hex}")
    with partial.open("xb") as destination:
        destination.write(header)
        destination.truncate(
            len(header) + exported_count * VERTEX_DTYPE.itemsize
        )

    native = np.memmap(
        source,
        dtype=native_dtype,
        mode="r",
        offset=source_offset,
        shape=(count,),
    )
    canonical = np.memmap(
        partial,
        dtype=VERTEX_DTYPE,
        mode="r+",
        offset=len(header),
        shape=(exported_count,),
    )
    target_start = 0
    for source_start in range(0, count, block_size):
        source_stop = min(count, source_start + block_size)
        first = source_start + (-source_start % stride)
        if first >= source_stop:
            continue
        selected = native[first:source_stop:stride]
        target_stop = target_start + len(selected)
        positions = np.column_stack(
            (selected["x"], selected["y"], selected["z"])
        ).astype(np.float64)
        normals = np.column_stack(
            (selected["nx"], selected["ny"], selected["nz"])
        ).astype(np.float64)
        transformed = positions @ frame
        transformed_normals = normals @ frame
        canonical["x"][target_start:target_stop] = transformed[:, 0]
        canonical["y"][target_start:target_stop] = transformed[:, 1]
        canonical["z"][target_start:target_stop] = transformed[:, 2]
        canonical["nx"][target_start:target_stop] = transformed_normals[:, 0]
        canonical["ny"][target_start:target_stop] = transformed_normals[:, 1]
        canonical["nz"][target_start:target_stop] = transformed_normals[:, 2]
        for channel in ("red", "green", "blue", "classification"):
            canonical[channel][target_start:target_stop] = selected[channel]
        canonical["source_index"][target_start:target_stop] = np.arange(
            first,
            source_stop,
            stride,
            dtype=np.uint32,
        )
        target_start = target_stop
    if target_start != exported_count:
        raise RuntimeError(
            f"canonical count mismatch: expected {exported_count}, "
            f"wrote {target_start}"
        )
    canonical.flush()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite canonical export: {output}"
        )
    partial.rename(output)
    return {
        "source": str(source),
        "output": str(output),
        "stride": stride,
        "source_point_count": count,
        "exported_point_count": exported_count,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "coordinate_frame": coordinate_frame,
    }
