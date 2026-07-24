from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from plant_cleanup.plyio import read_cloud


WEB_DTYPE = np.dtype(
    [
        ("position", "<f4", (3,)),
        ("color", "u1", (4,)),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_web_preview(
    source_path: Path, output_path: Path, *, max_points: int = 1_200_000
) -> dict[str, Any]:
    """Write deterministic interleaved XYZ/RGBA records for the WebGL viewer."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    cloud = read_cloud(source_path.resolve())
    stride = max(1, math.ceil(len(cloud) / max_points))
    sampled = cloud[::stride]
    preview = np.empty(len(sampled), dtype=WEB_DTYPE)
    preview["position"][:, 0] = sampled["x"]
    preview["position"][:, 1] = sampled["y"]
    preview["position"][:, 2] = sampled["z"]
    preview["color"][:, 0] = sampled["red"]
    preview["color"][:, 1] = sampled["green"]
    preview["color"][:, 2] = sampled["blue"]
    preview["color"][:, 3] = 255
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.tofile(output_path)
    positions = preview["position"]
    bounds = (
        {
            "min": positions.min(axis=0).astype(float).tolist(),
            "max": positions.max(axis=0).astype(float).tolist(),
        }
        if len(positions)
        else None
    )
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path),
        "source_point_count": int(len(cloud)),
        "preview_point_count": int(len(preview)),
        "stride": stride,
        "record_bytes": int(WEB_DTYPE.itemsize),
        "bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
        "bounds": bounds,
    }
