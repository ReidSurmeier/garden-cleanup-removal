from __future__ import annotations

import numpy as np
import pytest

from plant_cleanup.cloud_render import _sample_cloud_for_render


def test_proof_render_sampling_is_bounded_and_preserves_endpoints() -> None:
    cloud = np.arange(31)

    sampled, step = _sample_cloud_for_render(cloud, max_points=7)

    assert sampled.tolist() == [0, 5, 10, 15, 20, 25, 30]
    assert step == pytest.approx(5.0)


def test_proof_render_sampling_keeps_small_cloud_unchanged() -> None:
    cloud = np.arange(5)

    sampled, step = _sample_cloud_for_render(cloud, max_points=7)

    assert sampled is cloud
    assert step == 1.0


def test_proof_render_sampling_supports_one_point_limit() -> None:
    sampled, step = _sample_cloud_for_render(np.arange(5), max_points=1)

    assert sampled.tolist() == [2]
    assert step == 5.0


def test_proof_render_sampling_rejects_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="max_points must be positive"):
        _sample_cloud_for_render(np.arange(5), max_points=0)
