from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plant_cleanup.plyio import VERTEX_DTYPE
from plant_cleanup.source_photo_cleanup import run_source_photo_cleanup


def test_source_photo_cleanup_writes_publishable_additive_artifacts(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    baseline = tmp_path / "baseline"
    semantic = tmp_path / "source-photo"
    output = tmp_path / "output"
    (baseline / "semantic").mkdir(parents=True)
    (baseline / "vision-sam2").mkdir()
    (baseline / "dense-semantic").mkdir()
    (baseline / "final").mkdir()
    semantic.mkdir()
    source = tmp_path / "source.ply"
    source.write_bytes(b"immutable source")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "profile": {"vertical_span": 10.0},
                "dense_semantic": {"propagation": {}},
                "color_correction": {},
                "proof_render_size": 32,
            }
        ),
        encoding="utf-8",
    )
    (baseline / "run-report.json").write_text(
        json.dumps(
            {
                "source_opened_read_only": True,
                "source": str(source),
                "config": str(config),
                "models": {},
                "semantic": {
                    "support_plane": {
                        "coefficients": [0.0, 0.0, 0.0]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (semantic / "report.json").write_text(
        json.dumps(
            {
                "project_opened_read_only": True,
                "model": "oneformer",
                "camera_count": 12,
            }
        ),
        encoding="utf-8",
    )
    for path, values in (
        (
            baseline / "semantic" / "decision-codes.npy",
            np.array([1, 2, 3], dtype=np.uint8),
        ),
        (
            baseline / "vision-sam2" / "plant-votes.npy",
            np.array([1, 1, 1], dtype=np.uint8),
        ),
        (
            baseline / "vision-sam2" / "planter-votes.npy",
            np.array([0, 0, 0], dtype=np.uint8),
        ),
        (
            baseline / "dense-semantic" / "plant-votes.npy",
            np.array([1, 0, 1], dtype=np.uint8),
        ),
        (
            baseline / "dense-semantic" / "background-votes.npy",
            np.array([0, 1, 0], dtype=np.uint8),
        ),
        (
            semantic / "plant-votes.npy",
            np.array([2, 0, 2], dtype=np.uint8),
        ),
        (
            semantic / "background-votes.npy",
            np.array([0, 2, 0], dtype=np.uint8),
        ),
    ):
        np.save(path, values)
    previous = baseline / "final" / "plant-cleaned.ply"
    previous.write_bytes(b"previous")
    cloud = np.zeros(3, dtype=VERTEX_DTYPE)
    cloud["source_index"] = np.arange(3)

    module = "plant_cleanup.source_photo_cleanup"
    monkeypatch.setattr(f"{module}.read_cloud", lambda path: cloud)
    monkeypatch.setattr(
        f"{module}.remove_uncertain_floor",
        lambda *args, **kwargs: (
            np.ones(3, dtype=bool),
            {"source_opened_read_only": True},
        ),
    )
    monkeypatch.setattr(
        f"{module}.propagate_dense_semantic_evidence",
        lambda *args, **kwargs: (
            np.array([True, False, True]),
            np.ones(3, dtype=bool),
            {"status": "complete"},
        ),
    )

    def write_cloud(
        cloud_value: np.ndarray,
        path: Path,
        mask: np.ndarray,
        decisions: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([int(mask.sum())]))

    monkeypatch.setattr(f"{module}.write_decision_cloud", write_cloud)

    def color(source_path: Path, output_path: Path, params: object) -> dict:
        output_path.write_bytes(source_path.read_bytes())
        return {"geometry_identity_preserved": True}

    monkeypatch.setattr(f"{module}.correct_cloud_colors", color)

    def render(path: Path, render_dir: Path, **kwargs: object) -> dict:
        render_dir.mkdir()
        for view in ("front", "side", "top"):
            (render_dir / f"{view}-rgb.png").write_bytes(b"png")
        return {"source": str(path)}

    monkeypatch.setattr(f"{module}.render_cloud_views", render)

    def viewer(**kwargs: object) -> dict:
        viewer_root = Path(kwargs["output"])
        viewer_root.mkdir(parents=True)
        (viewer_root / "viewer.html").write_text("viewer", encoding="utf-8")
        return {
            "layers": {
                name: {"preview_point_count": 1}
                for name in (
                    "source",
                    "previous",
                    "plant",
                    "conservative",
                    "rejected",
                    "uncertain",
                )
            }
        }

    monkeypatch.setattr(f"{module}._build_viewer", viewer)

    report = run_source_photo_cleanup(baseline, semantic, output)

    assert report["source_opened_read_only"]
    assert report["counts"]["plant_cleaned"] == 2
    assert (
        output / "final" / "plant-cleaned-color-corrected.ply"
    ).is_file()
    assert (
        output / "final" / "render-source" / "front-rgb.png"
    ).is_file()
    assert (output / "review" / "viewer.html").is_file()
    assert (output / "run-report.json").is_file()
    assert source.read_bytes() == b"immutable source"
