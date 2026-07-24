from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from plant_cleanup.plyio import read_cloud


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lowest_significant_support(
    z: np.ndarray,
    normals_z: np.ndarray,
    excess_green: np.ndarray,
    *,
    robust_low: float,
    robust_high: float,
    normal_min: float,
    vegetation_excess_green_min: float,
    bins: int = 160,
) -> tuple[float, dict[str, Any]]:
    candidate = (
        (np.abs(normals_z) >= normal_min)
        & (excess_green < vegetation_excess_green_min)
        & (z >= robust_low)
        & (z <= robust_high)
    )
    heights = z[candidate]
    if len(heights) < 100:
        raise ValueError("too few horizontal non-vegetation points for support")
    counts, edges = np.histogram(
        heights,
        bins=bins,
        range=(robust_low, robust_high),
    )
    smooth = np.convolve(
        counts,
        np.ones(5, dtype=np.float64) / 5.0,
        mode="same",
    )
    padded = np.pad(smooth, (1, 1), constant_values=0.0)
    padded_peaks, _ = find_peaks(
        padded,
        distance=4,
        prominence=max(float(smooth.max()) * 0.03, 1.0),
    )
    peaks = padded_peaks - 1
    peaks = peaks[(peaks >= 0) & (peaks < len(smooth))]
    significant = peaks[smooth[peaks] >= float(smooth.max()) * 0.12]
    if len(significant):
        selected = int(significant[0])
        strategy = "lowest_significant_smoothed_mode"
    else:
        selected = int(np.argmax(smooth))
        strategy = "global_smoothed_mode_fallback"
    center = (edges[selected] + edges[selected + 1]) / 2.0
    half_window = 2.5 * (edges[1] - edges[0])
    local = heights[np.abs(heights - center) <= half_window]
    support_height = float(np.median(local)) if len(local) else float(center)
    return support_height, {
        "strategy": strategy,
        "candidate_points": int(len(heights)),
        "histogram_bins": bins,
        "selected_bin": selected,
        "selected_smoothed_count": float(smooth[selected]),
        "significant_peak_count": int(len(significant)),
    }


def build_adaptive_config(source_path: Path, base_config: dict) -> dict:
    """Derive a source-bound cleanup profile without changing the cloud."""

    source_path = source_path.resolve()
    cloud = read_cloud(source_path)
    z = np.asarray(cloud["z"], dtype=np.float64)
    robust_low, robust_high = np.percentile(z, [0.5, 99.5])
    vertical_span = float(robust_high - robust_low)
    if not np.isfinite(vertical_span) or vertical_span <= 0:
        raise ValueError("source cloud has no usable vertical span")
    rgb = np.column_stack(
        (cloud["red"], cloud["green"], cloud["blue"])
    ).astype(np.float64)
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]

    config = copy.deepcopy(base_config)
    support = config["support_estimation"]
    support_height, support_report = _lowest_significant_support(
        z,
        np.asarray(cloud["nz"], dtype=np.float64),
        excess_green,
        robust_low=float(robust_low),
        robust_high=float(robust_high),
        normal_min=float(support["support_normal_min"]),
        vegetation_excess_green_min=float(
            support["vegetation_excess_green_min"]
        ),
    )
    target = config["target_classification"]
    target.update(
        {
            "voxel_size": vertical_span / 150.0,
            "plant_z_min": support_height + vertical_span * 0.04,
            "plant_z_max": float(robust_high + vertical_span * 0.02),
            "seed_z_min": support_height + vertical_span * 0.04,
            "seed_z_max": float(robust_high + vertical_span * 0.02),
            "min_component_seed_points": max(
                500,
                int(round(len(cloud) * 0.0004)),
            ),
            "max_target_components": 6,
            "preservation_margin": vertical_span * 0.04,
        }
    )
    for key in (
        "focus_x_min",
        "focus_x_max",
        "focus_y_min",
        "focus_y_max",
    ):
        target.pop(key, None)

    support.update(
        {
            "support_bin_size": vertical_span / 160.0,
            "support_clearance": vertical_span * 0.04,
            "support_height": support_height,
            "strategy": "adaptive-lowest-significant-mode-v1",
        }
    )
    config["semantic_refinement"].update(
        {
            "height_band": vertical_span * 0.12,
            "growth_radius": vertical_span * 0.012,
        }
    )
    config["method"] = {
        "name": (
            "scale-aware target components with competing multi-view "
            "semantic evidence"
        ),
        "version": "root-safe-v2-adaptive",
    }
    config["dense_semantic"] = {
        "model": "shi-labs/oneformer_ade20k_swin_large",
        "plant_labels": [
            "plant",
            "tree",
            "grass",
            "flower",
            "palm, palm tree"
        ],
        "background_labels": ["*"],
        "propagation": {
            "conservative_background_factor": 2.0,
            "strict_background_factor": 10.0,
            "ground_height_fraction": 0.05,
            "ground_normal_min": 0.6,
            "structural_protection_radius_fraction": 0.05,
            "minimum_seed_points": 50
        }
    }
    config["profile"] = {
        "source": source_path.name,
        "source_sha256": _sha256(source_path),
        "source_point_count": int(len(cloud)),
        "robust_z_percentiles": [
            float(robust_low),
            float(robust_high),
        ],
        "vertical_span": vertical_span,
        "support": support_report,
    }
    return config
