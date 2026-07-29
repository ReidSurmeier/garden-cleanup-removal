from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from plant_cleanup.plyio import VERTEX_DTYPE, read_cloud


@dataclass(frozen=True)
class NormalizationParameters:
    assumed_camera_height_m: float = 1.8
    minimum_ground_points: int = 100
    minimum_aligned_cameras: int = 3
    maximum_up_disagreement_degrees: float = 20.0

    def __post_init__(self) -> None:
        if self.assumed_camera_height_m <= 0:
            raise ValueError("assumed camera height must be positive")
        if self.minimum_ground_points < 3:
            raise ValueError("at least three ground points are required")
        if self.minimum_aligned_cameras < 1:
            raise ValueError("at least one aligned camera is required")
        if not 0 < self.maximum_up_disagreement_degrees < 90:
            raise ValueError("maximum up disagreement must be between 0 and 90")


def _normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError("cannot normalize a zero-length or non-finite vector")
    return vector / length


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    center = np.mean(points, axis=0, dtype=np.float64)
    centered = points - center
    scatter = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    normal = _normalized(eigenvectors[:, 0])
    offset = -float(normal @ center)
    return normal, offset, eigenvalues


def _recorded_support_plane(
    report: dict[str, Any],
    *,
    minimum_candidate_points: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if report.get("strategy") != "median_low_support_normals":
        raise ValueError("support plane does not contain a measured normal")
    coefficients = np.asarray(report.get("coefficients"), dtype=np.float64)
    recorded_normal = np.asarray(report.get("normal"), dtype=np.float64)
    if coefficients.shape != (3,) or recorded_normal.shape != (3,):
        raise ValueError("support plane must contain 3D coefficients and normal")
    if not np.all(np.isfinite(coefficients)) or not np.all(
        np.isfinite(recorded_normal)
    ):
        raise ValueError("support plane evidence must be finite")
    normal_candidate_points = int(report.get("normal_candidate_points", 0))
    offset_candidate_points = int(report.get("offset_candidate_points", 0))
    if (
        normal_candidate_points < minimum_candidate_points
        or offset_candidate_points < minimum_candidate_points
    ):
        raise ValueError("support plane has too few measured candidate points")

    a, b, c = coefficients
    coefficient_normal = _normalized(np.array((-a, -b, 1.0)))
    recorded_normal = _normalized(recorded_normal)
    if float(coefficient_normal @ recorded_normal) < 1.0 - 1e-5:
        raise ValueError("support plane normal conflicts with its coefficients")
    ground_offset = -float(c * coefficient_normal[2])
    return (
        coefficient_normal,
        ground_offset,
        {
            "strategy": str(report["strategy"]),
            "normal_candidate_points": normal_candidate_points,
            "offset_candidate_points": offset_candidate_points,
        },
    )


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalized(source)
    target = _normalized(target)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1e-12:
        helper = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        if abs(float(source @ helper)) > 0.9:
            helper = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        axis = _normalized(np.cross(source, helper))
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)
    cross = np.cross(source, target)
    skew = np.array(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ),
        dtype=np.float64,
    )
    return np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)


def apply_similarity_transform(
    coordinates: np.ndarray,
    matrix: list[list[float]] | np.ndarray,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (point_count, 3)")
    if transform.shape != (4, 4):
        raise ValueError("normalization matrix must have shape (4, 4)")
    homogeneous = np.column_stack(
        (coordinates, np.ones(len(coordinates), dtype=np.float64))
    )
    return (homogeneous @ transform.T)[:, :3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _similarity_rotation(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    scales = np.linalg.norm(linear, axis=0)
    scale = float(np.mean(scales))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalization matrix scale must be positive")
    if not np.allclose(scales, scale, rtol=1e-6, atol=1e-9):
        raise ValueError("normalization matrix must use uniform scale")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("normalization matrix rotation must be orthonormal")
    if np.linalg.det(rotation) < 0.999999:
        raise ValueError("normalization matrix must preserve handedness")
    return scale, rotation


def write_normalized_cloud(
    source_path: Path,
    output_path: Path,
    matrix: list[list[float]] | np.ndarray,
    *,
    chunk_size: int = 1_000_000,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    transform = np.asarray(matrix, dtype=np.float64)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite normalized cloud: {output_path}")
    if transform.shape != (4, 4):
        raise ValueError("normalization matrix must have shape (4, 4)")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError("normalization matrix must be affine")
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    scale, rotation = _similarity_rotation(transform)
    source = read_cloud(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(
        f"{output_path.name}.partial-{uuid4().hex}"
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment normalized from immutable source coordinates\n"
        f"element vertex {len(source)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar classification\n"
        "property uint source_index\n"
        "end_header\n"
    ).encode("ascii")
    try:
        with partial.open("xb") as destination:
            destination.write(header)
            for start in range(0, len(source), chunk_size):
                stop = min(start + chunk_size, len(source))
                selected = np.array(
                    source[start:stop],
                    dtype=VERTEX_DTYPE,
                    copy=True,
                )
                coordinates = np.column_stack(
                    (selected["x"], selected["y"], selected["z"])
                )
                normalized = apply_similarity_transform(
                    coordinates,
                    transform,
                )
                selected["x"], selected["y"], selected["z"] = normalized.T
                normals = np.column_stack(
                    (selected["nx"], selected["ny"], selected["nz"])
                ).astype(np.float64)
                transformed_normals = normals @ rotation.T
                lengths = np.linalg.norm(transformed_normals, axis=1)
                nonzero = lengths > 1e-12
                transformed_normals[nonzero] /= lengths[nonzero, None]
                transformed_normals[~nonzero] = 0.0
                selected["nx"], selected["ny"], selected["nz"] = (
                    transformed_normals.T
                )
                destination.write(selected.tobytes())
        if output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite normalized cloud: {output_path}"
            )
        partial.rename(output_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "normalized": str(output_path),
        "normalized_sha256": _sha256(output_path),
        "source_point_count": int(len(source)),
        "normalized_point_count": int(len(source)),
        "source_identity_preserved": True,
        "uniform_scale": scale,
    }


def camera_evidence_from_inventory(
    inventory: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not bool(inventory.get("project_opened_read_only")):
        raise ValueError("camera inventory lacks read-only provenance")
    frame = inventory.get("coordinate_frame")
    if not isinstance(frame, dict) or not frame.get("source"):
        raise ValueError("camera inventory lacks source coordinate frame")
    cameras = inventory.get("cameras")
    if not isinstance(cameras, list):
        raise ValueError("camera inventory cameras must be a list")
    centers: list[list[float]] = []
    up_vectors: list[list[float]] = []
    for camera in cameras:
        if (
            not isinstance(camera, dict)
            or not bool(camera.get("enabled"))
            or not bool(camera.get("aligned"))
        ):
            continue
        center = camera.get("source_frame_center")
        up = camera.get("source_frame_up")
        if center is None or up is None:
            continue
        center_array = np.asarray(center, dtype=np.float64)
        up_array = np.asarray(up, dtype=np.float64)
        if center_array.shape != (3,) or up_array.shape != (3,):
            raise ValueError("camera source-frame evidence must be 3D")
        if not np.all(np.isfinite(center_array)) or not np.all(
            np.isfinite(up_array)
        ):
            raise ValueError("camera source-frame evidence must be finite")
        centers.append(center_array.tolist())
        up_vectors.append(_normalized(up_array).tolist())
    return (
        np.asarray(centers, dtype=np.float64).reshape((-1, 3)),
        np.asarray(up_vectors, dtype=np.float64).reshape((-1, 3)),
        {
            "inventory_camera_count": len(cameras),
            "usable_camera_count": len(centers),
            "coordinate_frame_source": str(frame["source"]),
        },
    )


def normalize_cleanup_layers(
    source_path: Path,
    layers: dict[str, Path],
    *,
    ground_mask: np.ndarray,
    camera_inventory: dict[str, Any],
    output_dir: Path,
    support_plane: dict[str, Any] | None = None,
    parameters: NormalizationParameters = NormalizationParameters(),
) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"normalization output already exists: {output_dir}"
        )
    if "source" in layers:
        raise ValueError("source is reserved as the immutable source layer")
    source = read_cloud(source_path)
    coordinates = np.column_stack(
        (source["x"], source["y"], source["z"])
    )
    camera_centers, camera_up_vectors, inventory_report = (
        camera_evidence_from_inventory(camera_inventory)
    )
    plan = estimate_normalization_plan(
        coordinates,
        ground_mask=ground_mask,
        camera_centers=camera_centers,
        camera_up_vectors=camera_up_vectors,
        support_plane=support_plane,
        parameters=parameters,
    )
    output_dir.mkdir(parents=True)
    layer_reports: dict[str, dict[str, Any]] = {}
    all_layers = {"source": source_path, **layers}
    for name, path in all_layers.items():
        if not name or any(character in name for character in "/\\"):
            raise ValueError(f"invalid normalization layer name: {name!r}")
        layer_reports[name] = write_normalized_cloud(
            path,
            output_dir / f"{name}-normalized.ply",
            plan["matrix"],
        )
    manifest = {
        "schema_version": 1,
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_point_count": int(len(source)),
        "source_opened_read_only": True,
        "plan": plan,
        "camera_inventory": inventory_report,
        "layers": layer_reports,
    }
    manifest_path = output_dir / "normalization-manifest.json"
    with manifest_path.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return {**manifest, "manifest": str(manifest_path)}


def estimate_normalization_plan(
    coordinates: np.ndarray,
    *,
    ground_mask: np.ndarray,
    camera_centers: np.ndarray,
    camera_up_vectors: np.ndarray,
    support_plane: dict[str, Any] | None = None,
    parameters: NormalizationParameters = NormalizationParameters(),
) -> dict[str, Any]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    ground_mask = np.asarray(ground_mask, dtype=bool)
    camera_centers = np.asarray(camera_centers, dtype=np.float64)
    camera_up_vectors = np.asarray(camera_up_vectors, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (point_count, 3)")
    if ground_mask.shape != (len(coordinates),):
        raise ValueError("ground mask must match the source point count")
    if camera_centers.ndim != 2 or camera_centers.shape[1] != 3:
        raise ValueError("camera centers must have shape (camera_count, 3)")
    if camera_up_vectors.shape != camera_centers.shape:
        raise ValueError("camera up vectors must match camera centers")

    ground = coordinates[ground_mask]
    if len(ground) < parameters.minimum_ground_points:
        raise ValueError(
            f"normalization requires at least "
            f"{parameters.minimum_ground_points} cleanup ground points"
        )
    if len(camera_centers) < parameters.minimum_aligned_cameras:
        raise ValueError(
            f"normalization requires at least "
            f"{parameters.minimum_aligned_cameras} aligned cameras"
        )

    fitted_normal, fitted_offset, eigenvalues = _fit_plane(ground)
    support_plane_report = None
    if support_plane is not None:
        ground_normal, ground_offset, support_plane_report = (
            _recorded_support_plane(
                support_plane,
                minimum_candidate_points=parameters.minimum_ground_points,
            )
        )
        orientation_basis = "cleanup_support_plane_report"
    else:
        ground_normal, ground_offset = fitted_normal, fitted_offset
        orientation_basis = "cleanup_ground_plane"
    normalized_camera_up = np.vstack(
        [_normalized(vector) for vector in camera_up_vectors]
    )
    camera_up = _normalized(np.sum(normalized_camera_up, axis=0))
    non_ground = coordinates[~ground_mask]
    if len(non_ground):
        sign_evidence = non_ground @ ground_normal + ground_offset
        normal_sign_basis = "non_ground_point_distribution"
    else:
        sign_evidence = camera_centers @ ground_normal + ground_offset
        normal_sign_basis = "camera_positions"
    if float(np.median(sign_evidence)) < 0:
        ground_normal = -ground_normal
        ground_offset = -ground_offset
    disagreement = float(
        np.degrees(
            np.arccos(
                np.clip(float(ground_normal @ camera_up), -1.0, 1.0)
            )
        )
    )

    raw_heights = camera_centers @ ground_normal + ground_offset
    positive_heights = raw_heights[raw_heights > 0]
    scale_resolved = (
        len(positive_heights) >= parameters.minimum_aligned_cameras
    )
    if scale_resolved:
        raw_camera_height = float(np.median(positive_heights))
        scale = parameters.assumed_camera_height_m / raw_camera_height
        raw_height_quartiles: list[float] | None = np.percentile(
            positive_heights,
            (25.0, 75.0),
        ).tolist()
    else:
        raw_camera_height = None
        raw_height_quartiles = None
        scale = 1.0
    rotation = _rotation_between(
        ground_normal,
        np.array((0.0, 0.0, 1.0), dtype=np.float64),
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[2, 3] = scale * ground_offset

    planarity = float(
        1.0 - eigenvalues[0] / max(float(eigenvalues.sum()), 1e-12)
    )
    strong_ground_evidence = planarity >= 0.99
    if support_plane_report is not None:
        automatic = (
            disagreement <= parameters.maximum_up_disagreement_degrees
        )
    else:
        automatic = strong_ground_evidence or (
            disagreement <= parameters.maximum_up_disagreement_degrees
            and planarity >= 0.9
        )
    automatic &= scale_resolved
    return {
        "schema_version": 1,
        "status": "automatic" if automatic else "needs_review",
        "matrix": matrix.tolist(),
        "rotation": rotation.tolist(),
        "scale": scale,
        "translation": matrix[:3, 3].tolist(),
        "evidence": {
            "orientation_basis": orientation_basis,
            "ground": {
                "candidate_point_count": int(len(ground)),
                "normal": ground_normal.tolist(),
                "offset": ground_offset,
                "planarity": planarity,
                "normal_sign_basis": normal_sign_basis,
                "recorded_support_plane": support_plane_report,
            },
            "cameras": {
                "aligned_camera_count": int(len(camera_centers)),
                "above_ground_camera_count": int(len(positive_heights)),
                "consensus_up": camera_up.tolist(),
                "up_disagreement_degrees": disagreement,
                "raw_height_median": raw_camera_height,
                "raw_height_quartiles": raw_height_quartiles,
                "assumed_camera_height_m": (
                    parameters.assumed_camera_height_m
                ),
            },
            "scale": {
                "status": "estimated" if scale_resolved else "unresolved",
                "basis": (
                    "assumed_camera_height"
                    if scale_resolved
                    else "identity_fallback_invalid_camera_geometry"
                ),
            },
        },
    }
