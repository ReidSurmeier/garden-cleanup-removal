from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any


RECORD = struct.Struct("<ffffffBBBBI")


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-12:
        raise ValueError("cannot normalize a zero-length coordinate axis")
    return tuple(
        0.0 if abs(value / length) < 1e-12 else value / length
        for value in vector
    )  # type: ignore[return-value]


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _camera_coordinate_frame(chunk: Any) -> dict[str, Any]:
    x_axes: list[tuple[float, float, float]] = []
    y_axes: list[tuple[float, float, float]] = []
    for camera in getattr(chunk, "cameras", []):
        if camera.transform is None:
            continue
        matrix = list(camera.transform)
        x_axes.append(_normalize((matrix[0], matrix[4], matrix[8])))
        y_axes.append(_normalize((matrix[1], matrix[5], matrix[9])))
    if not x_axes:
        return {
            "right": [1.0, 0.0, 0.0],
            "forward": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "source": "identity_fallback_no_aligned_cameras",
        }

    mean_x = tuple(sum(axis[index] for axis in x_axes) for index in range(3))
    mean_y = tuple(sum(axis[index] for axis in y_axes) for index in range(3))
    up = _normalize(tuple(-value for value in mean_y))
    horizontal_x = tuple(
        mean_x[index] - _dot(mean_x, up) * up[index]
        for index in range(3)
    )
    right = _normalize(horizontal_x)
    forward = _normalize(_cross(up, right))
    return {
        "right": list(right),
        "forward": list(forward),
        "up": list(up),
        "source": "mean_aligned_camera_axes",
    }


def _reframe(
    vector: Any,
    frame: dict[str, Any],
    *,
    normalize: bool = False,
) -> tuple[float, float, float]:
    source = (float(vector[0]), float(vector[1]), float(vector[2]))
    transformed = tuple(
        _dot(source, tuple(frame[axis]))
        for axis in ("right", "forward", "up")
    )
    return _normalize(transformed) if normalize else transformed  # type: ignore[arg-type, return-value]


def _channel(value: int | float) -> int:
    numeric = float(value)
    if 0.0 <= numeric <= 1.0 and not numeric.is_integer():
        numeric *= 255.0
    elif numeric > 255.0:
        numeric /= 257.0
    return max(0, min(255, round(numeric)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_reader_cloud_readonly(
    project: Path,
    output: Path,
    metashape: Any,
    *,
    stride: int = 1,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    """Stream a Metashape point cloud without modifying its source project."""

    if stride < 1 or chunk_size < 1:
        raise ValueError("stride and chunk_size must be positive")
    project = project.resolve()
    output = output.resolve()
    if not project.is_file():
        raise FileNotFoundError(project)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite export: {output}")
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

    coordinate_frame = _camera_coordinate_frame(chunk)
    source_count = int(cloud.point_count)
    exported_count = (source_count + stride - 1) // stride
    reader = metashape.PointCloud.Reader()
    reader.open(cloud)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment source_index is the sequential Metashape reader index\n"
        f"comment coordinate_frame {coordinate_frame['source']}\n"
        f"element vertex {exported_count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar classification\n"
        "property uint source_index\n"
        "end_header\n"
    ).encode("ascii")

    source_index = 0
    written = 0
    try:
        with output.open("xb") as destination:
            destination.write(header)
            while True:
                points = reader.read(chunk_size)
                if not points:
                    break
                buffer = bytearray()
                for point in points:
                    current_index = source_index
                    source_index += 1
                    if current_index % stride:
                        continue
                    position = _reframe(point.position, coordinate_frame)
                    normal = (
                        _reframe(
                            point.normal,
                            coordinate_frame,
                            normalize=True,
                        )
                        if point.normal is not None
                        else (0.0, 0.0, 0.0)
                    )
                    color = (
                        point.color
                        if point.color is not None
                        else (0, 0, 0)
                    )
                    buffer.extend(
                        RECORD.pack(
                            float(position[0]),
                            float(position[1]),
                            float(position[2]),
                            float(normal[0]),
                            float(normal[1]),
                            float(normal[2]),
                            _channel(color[0]),
                            _channel(color[1]),
                            _channel(color[2]),
                            max(
                                0,
                                min(255, int(point.classification)),
                            ),
                            current_index,
                        )
                    )
                    written += 1
                destination.write(buffer)
    except Exception:
        output.unlink(missing_ok=True)
        raise

    if source_index != source_count or written != exported_count:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"reader count mismatch: source={source_count}/{source_index}, "
            f"exported={exported_count}/{written}"
        )
    return {
        "metashape_version": str(metashape.version),
        "project": str(project),
        "output": str(output),
        "read_only": True,
        "stride": stride,
        "source_point_count": source_count,
        "exported_point_count": written,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "coordinate_frame": coordinate_frame,
    }
