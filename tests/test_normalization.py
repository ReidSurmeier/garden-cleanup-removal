from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plant_cleanup.plyio import VERTEX_DTYPE, read_cloud
from railing_removal.normalization import (
    NormalizationParameters,
    apply_similarity_transform,
    camera_evidence_from_inventory,
    estimate_normalization_plan,
    normalize_cleanup_layers,
    write_normalized_cloud,
)


def _write_cloud(path: Path, points: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar classification\n"
        "property uint source_index\n"
        "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + points.tobytes())


def test_cleanup_ground_and_camera_evidence_produce_metric_z_up_plan() -> None:
    xs, ys = np.meshgrid(
        np.linspace(-4.0, 4.0, 17),
        np.linspace(-3.0, 3.0, 13),
    )
    ground = np.column_stack(
        (
            xs.ravel(),
            ys.ravel(),
            0.30 * xs.ravel() - 0.15 * ys.ravel() + 2.0,
        )
    )
    ground_normal = np.array((-0.30, 0.15, 1.0), dtype=np.float64)
    ground_normal /= np.linalg.norm(ground_normal)
    plant = ground[::20] + ground_normal * np.linspace(0.2, 2.5, 12)[:, None]
    coordinates = np.vstack((ground, plant))
    ground_mask = np.zeros(len(coordinates), dtype=bool)
    ground_mask[: len(ground)] = True

    camera_anchors = ground[::35]
    camera_centers = camera_anchors + ground_normal * 3.0
    camera_up_vectors = np.tile(ground_normal, (len(camera_centers), 1))

    plan = estimate_normalization_plan(
        coordinates,
        ground_mask=ground_mask,
        camera_centers=camera_centers,
        camera_up_vectors=camera_up_vectors,
        parameters=NormalizationParameters(assumed_camera_height_m=1.8),
    )

    normalized_ground = apply_similarity_transform(
        ground,
        plan["matrix"],
    )
    normalized_cameras = apply_similarity_transform(
        camera_centers,
        plan["matrix"],
    )

    assert plan["status"] == "automatic"
    assert plan["evidence"]["ground"]["candidate_point_count"] == len(ground)
    assert np.max(np.abs(normalized_ground[:, 2])) < 1e-6
    assert np.median(normalized_cameras[:, 2]) == pytest.approx(1.8, abs=1e-6)
    assert np.linalg.det(np.asarray(plan["rotation"])) > 0.999999


def test_coherent_cleanup_ground_overrides_tilted_camera_view_axis() -> None:
    xs, ys = np.meshgrid(
        np.linspace(-5.0, 5.0, 21),
        np.linspace(-4.0, 4.0, 17),
    )
    ground = np.column_stack(
        (
            xs.ravel(),
            ys.ravel(),
            0.70 * ys.ravel() - 3.0,
        )
    )
    normal = np.array((0.0, -0.70, 1.0), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    camera_centers = ground[::50] + 4.0 * normal
    camera_view_up = np.tile((0.0, 0.0, 1.0), (len(camera_centers), 1))

    plan = estimate_normalization_plan(
        ground,
        ground_mask=np.ones(len(ground), dtype=bool),
        camera_centers=camera_centers,
        camera_up_vectors=camera_view_up,
    )

    normalized_ground = apply_similarity_transform(ground, plan["matrix"])
    assert plan["evidence"]["cameras"]["up_disagreement_degrees"] > 20.0
    assert plan["evidence"]["orientation_basis"] == "cleanup_ground_plane"
    assert plan["status"] == "automatic"
    assert np.max(np.abs(normalized_ground[:, 2])) < 1e-6


def test_recorded_support_plane_overrides_contaminated_rejected_support() -> None:
    xs, ys = np.meshgrid(
        np.linspace(-5.0, 5.0, 21),
        np.linspace(-4.0, 4.0, 17),
    )
    floor = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))
    railing = np.column_stack(
        (
            np.linspace(-5.0, 5.0, 180),
            np.zeros(180),
            np.linspace(0.0, 8.0, 180),
        )
    )
    plant = np.column_stack(
        (
            np.zeros(40),
            np.ones(40),
            np.linspace(0.1, 5.0, 40),
        )
    )
    coordinates = np.vstack((floor, railing, plant))
    rejected_support = np.zeros(len(coordinates), dtype=bool)
    rejected_support[: len(floor) + len(railing)] = True
    camera_centers = np.array(
        (
            (-3.0, -3.0, 2.0),
            (0.0, -4.0, 2.0),
            (3.0, -3.0, 2.0),
        )
    )

    plan = estimate_normalization_plan(
        coordinates,
        ground_mask=rejected_support,
        camera_centers=camera_centers,
        camera_up_vectors=np.tile((0.0, 0.0, 1.0), (3, 1)),
        support_plane={
            "coefficients": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "normal_candidate_points": len(floor),
            "offset_candidate_points": len(floor),
            "strategy": "median_low_support_normals",
        },
    )

    normalized_floor = apply_similarity_transform(floor, plan["matrix"])
    assert plan["status"] == "automatic"
    assert plan["evidence"]["orientation_basis"] == (
        "cleanup_support_plane_report"
    )
    assert np.max(np.abs(normalized_floor[:, 2])) < 1e-6
    np.testing.assert_allclose(
        plan["evidence"]["ground"]["normal"],
        (0.0, 0.0, 1.0),
    )


def test_recorded_support_plane_disagreement_requires_review() -> None:
    xs, ys = np.meshgrid(
        np.linspace(-5.0, 5.0, 21),
        np.linspace(-4.0, 4.0, 17),
    )
    floor = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))
    tilted_normal = np.array((0.0, -0.5, 1.0), dtype=np.float64)
    tilted_normal /= np.linalg.norm(tilted_normal)
    camera_centers = np.array(
        (
            (-3.0, -3.0, 2.0),
            (0.0, -4.0, 2.0),
            (3.0, -3.0, 2.0),
        )
    )

    plan = estimate_normalization_plan(
        floor,
        ground_mask=np.ones(len(floor), dtype=bool),
        camera_centers=camera_centers,
        camera_up_vectors=np.tile((0.0, 0.0, 1.0), (3, 1)),
        support_plane={
            "coefficients": [0.0, 0.5, 0.0],
            "normal": tilted_normal.tolist(),
            "normal_candidate_points": len(floor),
            "offset_candidate_points": len(floor),
            "strategy": "median_low_support_normals",
        },
    )

    assert plan["evidence"]["cameras"]["up_disagreement_degrees"] > 20.0
    assert plan["status"] == "needs_review"


def test_camera_positions_resolve_ground_side_when_view_axis_is_inverted() -> None:
    xs, ys = np.meshgrid(
        np.linspace(-3.0, 3.0, 17),
        np.linspace(-2.0, 2.0, 13),
    )
    ground = np.column_stack(
        (
            xs.ravel(),
            ys.ravel(),
            0.25 * xs.ravel() + 1.0,
        )
    )
    normal = np.array((-0.25, 0.0, 1.0), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    camera_centers = ground[::35] + 3.0 * normal
    inverted_view_up = np.tile(-normal, (len(camera_centers), 1))

    plan = estimate_normalization_plan(
        ground,
        ground_mask=np.ones(len(ground), dtype=bool),
        camera_centers=camera_centers,
        camera_up_vectors=inverted_view_up,
    )

    normalized_cameras = apply_similarity_transform(
        camera_centers,
        plan["matrix"],
    )
    assert plan["status"] == "automatic"
    assert np.all(normalized_cameras[:, 2] > 0.0)
    assert plan["evidence"]["ground"]["normal_sign_basis"] == "camera_positions"


def test_point_mass_resolves_up_when_camera_height_is_inconsistent() -> None:
    xs, ys = np.meshgrid(
        np.linspace(-3.0, 3.0, 17),
        np.linspace(-2.0, 2.0, 13),
    )
    ground = np.column_stack((xs.ravel(), ys.ravel(), np.ones(xs.size)))
    plant = ground[::20] + np.column_stack(
        (
            np.zeros(12),
            np.zeros(12),
            np.linspace(0.2, 3.0, 12),
        )
    )
    coordinates = np.vstack((ground, plant))
    ground_mask = np.zeros(len(coordinates), dtype=bool)
    ground_mask[: len(ground)] = True
    inconsistent_camera_centers = ground[::35] - np.array((0.0, 0.0, 2.0))

    plan = estimate_normalization_plan(
        coordinates,
        ground_mask=ground_mask,
        camera_centers=inconsistent_camera_centers,
        camera_up_vectors=np.tile(
            (0.0, 0.0, 1.0),
            (len(inconsistent_camera_centers), 1),
        ),
    )

    normalized = apply_similarity_transform(coordinates, plan["matrix"])
    assert plan["status"] == "needs_review"
    assert plan["scale"] == 1.0
    assert plan["evidence"]["ground"]["normal_sign_basis"] == (
        "non_ground_point_distribution"
    )
    assert plan["evidence"]["scale"]["status"] == "unresolved"
    assert np.max(np.abs(normalized[: len(ground), 2])) < 1e-6
    assert np.all(normalized[len(ground) :, 2] > 0.0)


def test_normalized_cloud_preserves_every_source_point_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ply"
    output = tmp_path / "normalized.ply"
    points = np.zeros(4, dtype=VERTEX_DTYPE)
    points["x"] = (0.0, 1.0, 2.0, 3.0)
    points["y"] = (2.0, 3.0, 4.0, 5.0)
    points["z"] = (-1.0, 0.0, 1.0, 2.0)
    points["nx"] = 1.0
    points["red"] = (10, 20, 30, 40)
    points["green"] = (50, 60, 70, 80)
    points["blue"] = (90, 100, 110, 120)
    points["classification"] = (1, 2, 3, 4)
    points["source_index"] = (900, 100, 700, 300)
    _write_cloud(source, points)
    matrix = np.array(
        (
            (0.0, -2.0, 0.0, 5.0),
            (2.0, 0.0, 0.0, -3.0),
            (0.0, 0.0, 2.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    report = write_normalized_cloud(source, output, matrix)
    normalized = read_cloud(output)

    assert report["source_point_count"] == len(points)
    assert report["normalized_point_count"] == len(points)
    np.testing.assert_array_equal(normalized["source_index"], points["source_index"])
    np.testing.assert_array_equal(
        normalized["classification"],
        points["classification"],
    )
    np.testing.assert_array_equal(normalized["red"], points["red"])
    expected = apply_similarity_transform(
        np.column_stack((points["x"], points["y"], points["z"])),
        matrix,
    )
    np.testing.assert_allclose(
        np.column_stack((normalized["x"], normalized["y"], normalized["z"])),
        expected,
    )
    np.testing.assert_allclose(
        np.column_stack((normalized["nx"], normalized["ny"], normalized["nz"])),
        np.tile((0.0, 1.0, 0.0), (len(points), 1)),
        atol=1e-7,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_normalized_cloud(source, output, matrix)


def test_camera_evidence_requires_read_only_source_frame_inventory() -> None:
    inventory = {
        "project_opened_read_only": True,
        "coordinate_frame": {"source": "mean_aligned_camera_axes"},
        "cameras": [
            {
                "enabled": True,
                "aligned": True,
                "source_frame_center": [1.0, 2.0, 3.0],
                "source_frame_up": [0.0, 0.0, 1.0],
            },
            {
                "enabled": False,
                "aligned": True,
                "source_frame_center": [9.0, 9.0, 9.0],
                "source_frame_up": [0.0, 0.0, 1.0],
            },
            {
                "enabled": True,
                "aligned": False,
                "source_frame_center": None,
                "source_frame_up": None,
            },
        ],
    }

    centers, up_vectors, report = camera_evidence_from_inventory(inventory)

    np.testing.assert_allclose(centers, ((1.0, 2.0, 3.0),))
    np.testing.assert_allclose(up_vectors, ((0.0, 0.0, 1.0),))
    assert report == {
        "inventory_camera_count": 3,
        "usable_camera_count": 1,
        "coordinate_frame_source": "mean_aligned_camera_axes",
    }
    inventory["project_opened_read_only"] = False
    with pytest.raises(ValueError, match="read-only provenance"):
        camera_evidence_from_inventory(inventory)


def test_cleanup_layers_share_one_transform_and_one_source_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ply"
    plant = tmp_path / "plant.ply"
    rejected = tmp_path / "rejected.ply"
    points = np.zeros(130, dtype=VERTEX_DTYPE)
    points["x"] = np.linspace(-3.0, 3.0, len(points))
    points["y"] = np.tile(np.linspace(-2.0, 2.0, 13), 10)
    points["z"] = 0.2 * points["x"] - 0.1 * points["y"] + 1.0
    points["nz"] = 1.0
    points["red"] = 25
    points["green"] = 100
    points["blue"] = 40
    points["source_index"] = np.arange(len(points)) * 11
    points["classification"][:100] = 4
    points["classification"][100:] = 1
    _write_cloud(source, points)
    _write_cloud(plant, points[100:])
    _write_cloud(rejected, points[:100])
    normal = np.array((-0.2, 0.1, 1.0))
    normal /= np.linalg.norm(normal)
    camera_centers = points[::30]
    camera_xyz = np.column_stack(
        (camera_centers["x"], camera_centers["y"], camera_centers["z"])
    ) + 2.5 * normal
    inventory = {
        "project_opened_read_only": True,
        "coordinate_frame": {"source": "mean_aligned_camera_axes"},
        "cameras": [
            {
                "enabled": True,
                "aligned": True,
                "source_frame_center": center.tolist(),
                "source_frame_up": normal.tolist(),
            }
            for center in camera_xyz
        ],
    }
    ground_mask = np.zeros(len(points), dtype=bool)
    ground_mask[:100] = True

    report = normalize_cleanup_layers(
        source,
        {"plant": plant, "rejected": rejected},
        ground_mask=ground_mask,
        camera_inventory=inventory,
        output_dir=tmp_path / "normalized",
        parameters=NormalizationParameters(
            assumed_camera_height_m=1.75,
            minimum_aligned_cameras=3,
        ),
    )

    assert report["plan"]["status"] == "automatic"
    assert report["source_point_count"] == len(points)
    assert set(report["layers"]) == {"source", "plant", "rejected"}
    normalized_source = read_cloud(
        Path(report["layers"]["source"]["normalized"])
    )
    normalized_plant = read_cloud(
        Path(report["layers"]["plant"]["normalized"])
    )
    normalized_rejected = read_cloud(
        Path(report["layers"]["rejected"]["normalized"])
    )
    np.testing.assert_array_equal(
        np.sort(
            np.concatenate(
                (
                    normalized_plant["source_index"],
                    normalized_rejected["source_index"],
                )
            )
        ),
        np.sort(normalized_source["source_index"]),
    )
    assert Path(report["manifest"]).is_file()
    with pytest.raises(FileExistsError, match="output already exists"):
        normalize_cleanup_layers(
            source,
            {"plant": plant},
            ground_mask=ground_mask,
            camera_inventory=inventory,
            output_dir=tmp_path / "normalized",
        )
