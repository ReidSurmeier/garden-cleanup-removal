from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from plant_cleanup.adaptive_profile import build_adaptive_config
from plant_cleanup.plyio import VERTEX_DTYPE


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "2026-07-15-172629-stride8.json"


def _write_scaled_scene(path: Path, scale: float) -> None:
    rng = np.random.default_rng(42)
    lower_count = 4_000
    upper_count = 10_000
    plant_count = 8_000
    cloud = np.zeros(
        lower_count + upper_count + plant_count,
        dtype=VERTEX_DTYPE,
    )
    lower = slice(0, lower_count)
    upper = slice(lower_count, lower_count + upper_count)
    plant = slice(lower_count + upper_count, None)
    cloud["x"] = rng.uniform(-1, 1, len(cloud)) * scale
    cloud["y"] = rng.uniform(-1, 1, len(cloud)) * scale
    cloud["z"][lower] = (
        1.0 + rng.normal(0, 0.01, lower_count)
    ) * scale
    cloud["z"][upper] = (
        5.0 + rng.normal(0, 0.01, upper_count)
    ) * scale
    cloud["z"][plant] = rng.uniform(1.2, 8.0, plant_count) * scale
    cloud["nz"][lower] = 1.0
    cloud["nz"][upper] = 1.0
    cloud["nz"][plant] = rng.uniform(-0.5, 0.5, plant_count)
    cloud["red"] = 100
    cloud["green"] = 100
    cloud["blue"] = 100
    cloud["red"][plant] = 30
    cloud["green"][plant] = 150
    cloud["blue"][plant] = 40
    cloud["source_index"] = np.arange(len(cloud), dtype=np.uint32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(cloud)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar classification\n"
        "property uint source_index\nend_header\n"
    ).encode("ascii")
    path.write_bytes(header + cloud.tobytes())


def test_adaptive_config_is_scale_invariant_and_source_bound(
    tmp_path: Path,
) -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    small = tmp_path / "small.ply"
    large = tmp_path / "large.ply"
    _write_scaled_scene(small, 1.0)
    _write_scaled_scene(large, 10.0)
    before = hashlib.sha256(small.read_bytes()).hexdigest()

    small_config = build_adaptive_config(small, base)
    large_config = build_adaptive_config(large, base)

    assert small_config["profile"]["source_sha256"] == before
    assert hashlib.sha256(small.read_bytes()).hexdigest() == before
    assert small_config["support_estimation"]["support_height"] == pytest.approx(
        1.0,
        abs=0.1,
    )
    assert large_config["support_estimation"]["support_height"] == pytest.approx(
        10.0,
        abs=1.0,
    )
    for key in ("voxel_size", "preservation_margin"):
        ratio = (
            large_config["target_classification"][key]
            / small_config["target_classification"][key]
        )
        assert ratio == pytest.approx(10.0, rel=0.05)
    assert "focus_x_max" not in small_config["target_classification"]
    assert "focus_y_max" not in small_config["target_classification"]


def test_adaptive_config_falls_back_without_horizontal_support(
    tmp_path: Path,
) -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    source = tmp_path / "vertical-only.ply"
    _write_scaled_scene(source, 1.0)
    cloud = np.frombuffer(
        source.read_bytes().split(b"end_header\n", 1)[1],
        dtype=VERTEX_DTYPE,
    ).copy()
    cloud["nz"] = 0.0
    header = source.read_bytes().split(b"end_header\n", 1)[0]
    source.write_bytes(header + b"end_header\n" + cloud.tobytes())

    config = build_adaptive_config(source, base)

    assert config["profile"]["support"]["strategy"] == (
        "robust_low_percentile_fallback"
    )
    assert config["support_estimation"]["support_height"] == pytest.approx(
        config["profile"]["robust_z_percentiles"][0]
    )
