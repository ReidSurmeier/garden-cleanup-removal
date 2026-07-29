from __future__ import annotations

import numpy as np

from railing_removal.geometry_orientation import (
    estimate_axes_from_normal_pairs,
    estimate_rigid_axes_from_cloud,
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


def test_rigid_axis_estimator_ignores_neutral_pavement_normals() -> None:
    grid_x, grid_y = np.meshgrid(
        np.linspace(-5.0, 5.0, 50),
        np.linspace(-5.0, 5.0, 50),
    )
    ground_points = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size))
    )
    rng = np.random.default_rng(12)
    ground_normals = np.column_stack(
        (
            rng.normal(scale=0.18, size=grid_x.size),
            rng.normal(scale=0.18, size=grid_x.size),
            np.ones(grid_x.size),
        )
    )
    angles, heights = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False),
        np.linspace(0.0, 4.0, 20),
    )
    post_points = np.column_stack(
        (
            np.cos(angles.ravel()),
            np.sin(angles.ravel()),
            heights.ravel(),
        )
    )
    post_normals = np.column_stack(
        (
            np.cos(angles.ravel()),
            np.sin(angles.ravel()),
            np.zeros(angles.size),
        )
    )
    coordinates = np.vstack((ground_points, post_points))
    normals = np.vstack((ground_normals, post_normals))
    colors = np.full_like(coordinates, 128.0)

    result = estimate_rigid_axes_from_cloud(
        coordinates,
        normals,
        colors,
        reference_up=np.array((0.0, 0.0, 1.0)),
        maximum_sample_points=10_000,
    )

    assert result["status"] == "usable"
    assert result["vertical_surface_point_count"] == len(post_points)
    np.testing.assert_allclose(
        result["candidates"][0]["axis"],
        (0.0, 0.0, 1.0),
        atol=0.02,
    )
