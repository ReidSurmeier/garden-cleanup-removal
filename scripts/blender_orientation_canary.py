from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import bpy
from mathutils import Matrix, Vector
import numpy as np


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--maximum-points", type=int, default=80_000)
    parser.add_argument("--size", type=int, default=1200)
    return parser.parse_args(values)


def _rotation_to_z(up: list[float]) -> np.ndarray:
    source = np.asarray(up, dtype=np.float64)
    source /= np.linalg.norm(source)
    target = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if cosine > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1e-10:
        return np.diag((1.0, -1.0, -1.0))
    cross = np.cross(source, target)
    skew = np.array(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ),
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + cosine))


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.node_groups,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    emission_strength: float = 0.0,
    color_attribute: str | None = None,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.8
    if emission_strength:
        principled.inputs["Emission Color"].default_value = color
        principled.inputs["Emission Strength"].default_value = emission_strength
    if color_attribute:
        attribute = material.node_tree.nodes.new("ShaderNodeAttribute")
        attribute.attribute_name = color_attribute
        material.node_tree.links.new(
            attribute.outputs["Color"],
            principled.inputs["Base Color"],
        )
        material.node_tree.links.new(
            attribute.outputs["Color"],
            principled.inputs["Emission Color"],
        )
    return material


def _point_renderer(
    target: bpy.types.Object,
    *,
    radius: float,
    material: bpy.types.Material,
) -> None:
    modifier = target.modifiers.new("Point Renderer", "NODES")
    tree = bpy.data.node_groups.new("Point Renderer", "GeometryNodeTree")
    modifier.node_group = tree
    tree.interface.new_socket(
        name="Geometry",
        in_out="INPUT",
        socket_type="NodeSocketGeometry",
    )
    tree.interface.new_socket(
        name="Geometry",
        in_out="OUTPUT",
        socket_type="NodeSocketGeometry",
    )
    input_node = tree.nodes.new("NodeGroupInput")
    output_node = tree.nodes.new("NodeGroupOutput")
    points = tree.nodes.new("GeometryNodeMeshToPoints")
    points.mode = "VERTICES"
    points.inputs["Radius"].default_value = radius
    sphere = tree.nodes.new("GeometryNodeMeshIcoSphere")
    sphere.inputs["Radius"].default_value = radius
    sphere.inputs["Subdivisions"].default_value = 1
    instances = tree.nodes.new("GeometryNodeInstanceOnPoints")
    set_material = tree.nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    tree.links.new(input_node.outputs["Geometry"], points.inputs["Mesh"])
    tree.links.new(points.outputs["Points"], instances.inputs["Points"])
    tree.links.new(sphere.outputs["Mesh"], instances.inputs["Instance"])
    tree.links.new(instances.outputs["Instances"], set_material.inputs["Geometry"])
    tree.links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])


def _line_object(
    name: str,
    segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    *,
    width: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = width
    curve.bevel_resolution = 0
    for start, end in segments:
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (*start, 1.0)
        spline.points[1].co = (*end, 1.0)
    target = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(target)
    target.data.materials.append(material)
    return target


def _grid_and_axes(radius: float) -> None:
    grid = _material("Grid", (0.18, 0.2, 0.24, 1.0), emission_strength=0.3)
    red = _material("Axis X", (0.8, 0.08, 0.05, 1.0), emission_strength=1.0)
    green = _material("Axis Y", (0.05, 0.8, 0.12, 1.0), emission_strength=1.0)
    blue = _material("Axis Z", (0.05, 0.25, 1.0, 1.0), emission_strength=1.0)
    grid_segments = []
    for index in range(-10, 11):
        offset = radius * index / 10.0
        grid_segments.append(((-radius, offset, 0.0), (radius, offset, 0.0)))
        grid_segments.append(((offset, -radius, 0.0), (offset, radius, 0.0)))
    _line_object(
        "World XY Ground Grid",
        grid_segments,
        width=max(radius / 700.0, 1e-5),
        material=grid,
    )
    axis_width = max(radius / 220.0, 2e-5)
    _line_object("World X", [((0, 0, 0), (radius, 0, 0))], width=axis_width, material=red)
    _line_object("World Y", [((0, 0, 0), (0, radius, 0))], width=axis_width, material=green)
    _line_object("World Z", [((0, 0, 0), (0, 0, radius))], width=axis_width, material=blue)


def _camera(name: str) -> bpy.types.Object:
    data = bpy.data.cameras.new(name)
    data.type = "ORTHO"
    target = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(target)
    return target


def _aim(camera: bpy.types.Object, location: tuple[float, float, float], target: tuple[float, float, float]) -> None:
    camera.location = location
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _sample_mesh(target: bpy.types.Object, maximum_points: int) -> np.ndarray:
    count = len(target.data.vertices)
    coordinates = np.empty(count * 3, dtype=np.float32)
    target.data.vertices.foreach_get("co", coordinates)
    coordinates = coordinates.reshape((-1, 3)).astype(np.float64)
    if count <= maximum_points:
        return coordinates
    indices = np.linspace(0, count - 1, maximum_points).round().astype(np.int64)
    sampled = coordinates[indices]
    colors = None
    color_attribute = target.data.attributes.get("Col")
    if color_attribute is not None:
        raw_colors = np.empty(count * 4, dtype=np.float32)
        color_attribute.data.foreach_get("color", raw_colors)
        colors = raw_colors.reshape((-1, 4))[indices]
    mesh = bpy.data.meshes.new("Deterministic Point Sample")
    mesh.from_pydata(sampled.tolist(), [], [])
    if colors is not None:
        sampled_colors = mesh.attributes.new(
            name="Col",
            type="FLOAT_COLOR",
            domain="POINT",
        )
        sampled_colors.data.foreach_set("color", colors.ravel())
    target.data = mesh
    return sampled


def _configure_render(size: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.005, 0.006, 0.009)


def _render_views(
    output: Path,
    *,
    extent: float,
    z_center: float,
    scale_front: float,
    scale_side: float,
    scale_top: float,
) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    camera = _camera("Evaluation Camera")
    views = {
        "front": ((0.0, -extent * 3.0, z_center), scale_front),
        "side": ((extent * 3.0, 0.0, z_center), scale_side),
        "top": ((0.0, 0.0, extent * 3.0 + z_center), scale_top),
    }
    result: dict[str, str] = {}
    for name, (location, scale) in views.items():
        _aim(camera, location, (0.0, 0.0, z_center))
        camera.data.ortho_scale = max(scale, extent * 0.2)
        scene.camera = camera
        destination = output / f"{name}.png"
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        result[name] = str(destination)
    return result


def _evaluate_scan(
    item: dict[str, Any],
    output_root: Path,
    maximum_points: int,
    size: int,
) -> dict[str, Any]:
    _clear_scene()
    source = Path(item["source"]).resolve()
    bpy.ops.wm.ply_import(filepath=str(source))
    cloud = bpy.context.object
    cloud.name = "Plant Point Cloud"
    coordinates = _sample_mesh(cloud, maximum_points)
    candidates = item["candidates"]
    transformed: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    for candidate in candidates:
        identifier = str(candidate["candidate"])
        rotation = _rotation_to_z(candidate["up"])
        rotated = coordinates @ rotation.T
        center_xy = np.median(rotated[:, :2], axis=0)
        ground_z = float(np.quantile(rotated[:, 2], 0.01))
        rotated[:, :2] -= center_xy
        rotated[:, 2] -= ground_z
        transformed[identifier] = rotated
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:2, 3] = -center_xy
        matrix[2, 3] = -ground_z
        matrices[identifier] = matrix
    union = np.concatenate(list(transformed.values()), axis=0)
    lower = np.quantile(union, 0.005, axis=0)
    upper = np.quantile(union, 0.995, axis=0)
    spans = np.maximum(upper - lower, 1e-5)
    extent = float(max(spans.max(), 1e-3))
    grid_radius = float(max(spans[0], spans[1]) * 0.75)
    z_center = float((lower[2] + upper[2]) / 2.0)
    point_material = _material(
        "Plant",
        (0.3, 0.74, 0.24, 1.0),
        emission_strength=0.18,
        color_attribute="Col",
    )
    _point_renderer(
        cloud,
        radius=max(extent / 650.0, 1e-5),
        material=point_material,
    )
    _grid_and_axes(max(grid_radius, extent * 0.35))
    _configure_render(size)

    results = []
    for candidate in candidates:
        identifier = str(candidate["candidate"])
        matrix = matrices[identifier]
        cloud.matrix_world = Matrix(matrix.tolist())
        values = transformed[identifier]
        tolerance = max(extent * 0.005, 1e-5)
        rotation_angle = math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.asarray(candidate["up"], dtype=np.float64)[2],
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        candidate_output = output_root / item["scan_id"] / f"candidate-{identifier}"
        renders = _render_views(
            candidate_output,
            extent=extent,
            z_center=z_center,
            scale_front=max(spans[0], spans[2]) * 1.18,
            scale_side=max(spans[1], spans[2]) * 1.18,
            scale_top=max(spans[0], spans[1]) * 1.18,
        )
        results.append(
            {
                **candidate,
                "matrix": matrix.tolist(),
                "rotation_degrees": rotation_angle,
                "root_contact_tolerance": tolerance,
                "below_ground_fraction": float(np.mean(values[:, 2] < -tolerance)),
                "root_contact_fraction": float(np.mean(np.abs(values[:, 2]) <= tolerance)),
                "renders": renders,
            }
        )
    scan_output = output_root / item["scan_id"]
    blend_path = scan_output / "orientation-evaluation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    report = {
        "schema_version": 1,
        "scan_id": item["scan_id"],
        "source": str(source),
        "source_opened_read_only": True,
        "sample_point_count": int(len(coordinates)),
        "world_contract": {
            "up_axis": "+Z",
            "ground_plane": "XY at Z=0",
            "camera_projection": "orthographic",
            "candidate_camera_and_bounds_shared": True,
            "robust_ground_translation_quantile": 0.01,
        },
        "candidates": results,
        "blend_file": str(blend_path),
    }
    with (scan_output / "evaluation-report.json").open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return report


def main() -> None:
    args = _arguments()
    manifest = json.loads(
        args.manifest.resolve().read_text(encoding="utf-8-sig")
    )
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    reports = []
    for index, item in enumerate(manifest["scans"], start=1):
        print(
            f"[BLENDER-ORIENTATION] {index}/{len(manifest['scans'])} "
            f"{item['scan_id']}",
            flush=True,
        )
        reports.append(
            _evaluate_scan(
                item,
                output_root,
                args.maximum_points,
                args.size,
            )
        )
    batch = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "source_files_opened_read_only": True,
        "reports": [
            {
                "scan_id": report["scan_id"],
                "report": str(
                    output_root / report["scan_id"] / "evaluation-report.json"
                ),
            }
            for report in reports
        ],
    }
    with (output_root / "batch-report.json").open("x", encoding="utf-8") as destination:
        json.dump(batch, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(json.dumps(batch, indent=2), flush=True)


if __name__ == "__main__":
    main()
