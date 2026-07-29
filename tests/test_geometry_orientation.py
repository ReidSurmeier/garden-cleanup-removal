from __future__ import annotations

import numpy as np

from railing_removal.geometry_orientation import (
    estimate_axes_from_normal_pairs,
)


def test_cylindrical_surface_normals_recover_the_post_axis() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    left = np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros_like(angles))
    )
    right_angles = angles + np.radians(35.0)
    right = np.column_stack(
        (
            np.cos(right_angles),
            np.sin(right_angles),
            np.zeros_like(right_angles),
        )
    )
    rng = np.random.default_rng(42)
    outlier_left = rng.normal(size=(40, 3))
    outlier_right = rng.normal(size=(40, 3))

    result = estimate_axes_from_normal_pairs(
        np.vstack((left, outlier_left)),
        np.vstack((right, outlier_right)),
        minimum_support=100,
    )

    assert result["status"] == "usable"
    assert result["candidates"][0]["support_count"] >= 180
    np.testing.assert_allclose(
        np.abs(result["candidates"][0]["axis"]),
        (0.0, 0.0, 1.0),
        atol=1e-3,
    )


def test_unstructured_surface_normals_fail_closed() -> None:
    rng = np.random.default_rng(7)
    result = estimate_axes_from_normal_pairs(
        rng.normal(size=(30, 3)),
        rng.normal(size=(30, 3)),
        minimum_support=25,
        maximum_axis_agreement_degrees=5.0,
    )

    assert result["status"] == "insufficient_evidence"
