from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


VERTEX_DTYPE = np.dtype(
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
        ("source_index", "<u4"),
    ]
)


def read_cloud(path: Path) -> np.memmap:
    """Memory-map the binary PLY contract emitted by the Metashape reader adapter."""
    with path.open("rb") as source:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = source.readline()
            if not line:
                raise ValueError(f"missing PLY end_header in {path}")
            header.extend(line)
        offset = source.tell()
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("unsupported PLY format")
    count_line = next(
        (line for line in text.splitlines() if line.startswith("element vertex ")),
        None,
    )
    if count_line is None:
        raise ValueError("PLY vertex count is missing")
    count = int(count_line.rsplit(maxsplit=1)[1])
    expected_bytes = offset + count * VERTEX_DTYPE.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"PLY byte count mismatch: expected {expected_bytes}, got {path.stat().st_size}"
        )
    return np.memmap(path, dtype=VERTEX_DTYPE, mode="r", offset=offset, shape=(count,))


def summarize_cloud(path: Path) -> dict[str, Any]:
    cloud = read_cloud(path)
    coordinates = np.column_stack((cloud["x"], cloud["y"], cloud["z"]))
    rgb = np.column_stack((cloud["red"], cloud["green"], cloud["blue"])).astype(
        np.float64
    )
    classes, counts = np.unique(cloud["classification"], return_counts=True)
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    return {
        "point_count": int(len(cloud)),
        "bounds": {
            "min": coordinates.min(axis=0).astype(float).tolist(),
            "max": coordinates.max(axis=0).astype(float).tolist(),
        },
        "mean_rgb": rgb.mean(axis=0).tolist(),
        "mean_excess_green": float(excess_green.mean()),
        "excess_green_percentiles": {
            str(percentile): float(np.percentile(excess_green, percentile))
            for percentile in (1, 5, 25, 50, 75, 95, 99)
        },
        "classes": {
            str(int(classification)): int(count)
            for classification, count in zip(classes, counts, strict=True)
        },
        "source_index": {
            "min": int(cloud["source_index"].min()),
            "max": int(cloud["source_index"].max()),
        },
    }
