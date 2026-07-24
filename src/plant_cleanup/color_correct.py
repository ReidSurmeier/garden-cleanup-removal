from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plant_cleanup.plyio import VERTEX_DTYPE, read_cloud


@dataclass(frozen=True)
class ColorParameters:
    neutral_chroma_max: float = 0.18
    gain_min: float = 0.85
    gain_max: float = 1.15
    luminance_low_percentile: float = 1.0
    luminance_high_percentile: float = 99.0
    target_black: float = 6.0
    target_white: float = 246.0
    tone_scale_min: float = 0.85
    tone_scale_max: float = 1.15

    def __post_init__(self) -> None:
        if not 0 < self.neutral_chroma_max < 1:
            raise ValueError("neutral chroma threshold must be between zero and one")
        if not 0 < self.gain_min <= self.gain_max:
            raise ValueError("invalid color gain bounds")
        if not 0 < self.tone_scale_min <= self.tone_scale_max:
            raise ValueError("invalid tone scale bounds")


def _write_cloud(path: Path, cloud: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment colors corrected; geometry and source identity preserved\n"
        f"element vertex {len(cloud)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar classification\nproperty uint source_index\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as destination:
        destination.write(header)
        destination.write(np.asarray(cloud, dtype=VERTEX_DTYPE).tobytes())


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def correct_cloud_colors(
    source_path: Path,
    output_path: Path,
    parameters: ColorParameters = ColorParameters(),
) -> dict[str, Any]:
    """Apply bounded neutral-reference white balance and robust tone scaling."""
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    source = read_cloud(source_path)
    rgb = np.column_stack((source["red"], source["green"], source["blue"])).astype(
        np.float64
    )
    if len(source) == 0:
        corrected = np.array(source, dtype=VERTEX_DTYPE)
        _write_cloud(output_path, corrected)
        return {
            "source": str(source_path),
            "output": str(output_path),
            "point_count": 0,
            "quality_state": "unusable",
            "quality_reason": "empty_cloud",
            "neutral_reference_points": 0,
            "neutral_reference_rgb": [0.0, 0.0, 0.0],
            "channel_gains": [1.0, 1.0, 1.0],
            "luminance_before_percentiles": [],
            "luminance_after_percentiles": [],
            "geometry_identity_preserved": True,
            "parameters": asdict(parameters),
            "warnings": ["empty cloud; color correction skipped"],
        }
    maximum = rgb.max(axis=1)
    chroma = np.ptp(rgb, axis=1) / np.maximum(maximum, 1.0)
    luminance_before = _luminance(rgb)
    neutral = (
        (chroma <= parameters.neutral_chroma_max)
        & (luminance_before >= 16.0)
        & (luminance_before <= 240.0)
    )
    warnings: list[str] = []
    if int(neutral.sum()) >= 3:
        reference = np.median(rgb[neutral], axis=0)
        target = float(reference.mean())
        gains = np.clip(
            target / np.maximum(reference, 1.0),
            parameters.gain_min,
            parameters.gain_max,
        )
    else:
        reference = np.array([0.0, 0.0, 0.0])
        gains = np.ones(3, dtype=np.float64)
        warnings.append("too few neutral points; white-balance gains left at unity")

    balanced = np.clip(rgb * gains, 0.0, 255.0)
    luminance_balanced = _luminance(balanced)
    low, high = np.percentile(
        luminance_balanced,
        [parameters.luminance_low_percentile, parameters.luminance_high_percentile],
    )
    if high - low < 1e-6:
        tone_scale = np.ones(len(source), dtype=np.float64)
        warnings.append("luminance range collapsed; tone curve skipped")
    else:
        target_luminance = parameters.target_black + (
            np.clip((luminance_balanced - low) / (high - low), 0.0, 1.0)
            * (parameters.target_white - parameters.target_black)
        )
        tone_scale = np.clip(
            target_luminance / np.maximum(luminance_balanced, 1.0),
            parameters.tone_scale_min,
            parameters.tone_scale_max,
        )
    corrected_rgb = np.clip(
        np.rint(balanced * tone_scale[:, np.newaxis]), 0.0, 255.0
    ).astype(np.uint8)

    corrected = np.array(source, dtype=VERTEX_DTYPE)
    corrected["red"] = corrected_rgb[:, 0]
    corrected["green"] = corrected_rgb[:, 1]
    corrected["blue"] = corrected_rgb[:, 2]
    identity_fields = (
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "classification",
        "source_index",
    )
    identity_preserved = all(
        np.array_equal(source[field], corrected[field]) for field in identity_fields
    )
    if not identity_preserved:
        raise RuntimeError("color correction changed geometry or point identity")
    _write_cloud(output_path, corrected)
    luminance_after = _luminance(corrected_rgb.astype(np.float64))
    return {
        "source": str(source_path),
        "output": str(output_path),
        "point_count": int(len(source)),
        "quality_state": "usable",
        "quality_reason": None,
        "neutral_reference_points": int(neutral.sum()),
        "neutral_reference_rgb": reference.tolist(),
        "channel_gains": gains.tolist(),
        "luminance_before_percentiles": np.percentile(
            luminance_before, [1, 25, 50, 75, 99]
        ).tolist(),
        "luminance_after_percentiles": np.percentile(
            luminance_after, [1, 25, 50, 75, 99]
        ).tolist(),
        "geometry_identity_preserved": identity_preserved,
        "parameters": asdict(parameters),
        "warnings": warnings,
    }
