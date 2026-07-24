from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plant_cleanup.scene_evidence import fuse_scene_votes, run_scene_evidence


def test_scene_evidence_accepts_catalog_class_ids_with_underscores(
    tmp_path: Path,
) -> None:
    class_run = tmp_path / "class-run"
    class_run.mkdir()
    np.save(class_run / "plant-votes.npy", np.array([1, 0], dtype=np.uint8))
    np.save(class_run / "planter-votes.npy", np.array([0, 1], dtype=np.uint8))

    report = fuse_scene_votes(
        {"chain_barrier": class_run},
        tmp_path / "fused",
    )

    assert report["classes"]["chain_barrier"]["voted_point_count"] == 1
    assert (tmp_path / "fused" / "chain_barrier-votes.npy").is_file()


def test_missing_object_segmentation_is_recorded_without_failing_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_clipseg(
        cloud_path: Path,
        render_dir: Path,
        output_dir: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        output_dir.mkdir(parents=True)
        return {"views": []}

    def fake_sam2(
        cloud_path: Path,
        render_dir: Path,
        clipseg_dir: Path,
        output_dir: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        output_dir.mkdir(parents=True)
        np.save(
            output_dir / "plant-votes.npy",
            np.zeros(3, dtype=np.uint8),
        )
        np.save(
            output_dir / "planter-votes.npy",
            np.zeros(3, dtype=np.uint8),
        )
        return {"views": [{"status": "no-anchor"}]}

    monkeypatch.setattr(
        "plant_cleanup.scene_evidence.aggregate_clipseg_votes",
        fake_clipseg,
    )
    monkeypatch.setattr(
        "plant_cleanup.scene_evidence.aggregate_sam2_votes",
        fake_sam2,
    )

    report = run_scene_evidence(
        tmp_path / "source.ply",
        tmp_path / "renders",
        tmp_path / "scene",
        {
            "schema_version": 1,
            "scan_id": "scan",
            "plant_prompt": "complete plants",
            "classes": [
                {
                    "id": "railing",
                    "prompt": "rigid railing",
                    "distractor_prompt": "living stems",
                    "required_segmented_views": 1,
                }
            ],
        },
        clipseg_predictor=object(),
        sam2_predictor=object(),
    )

    assert report["classes"]["railing"]["segmented_views"] == 0
    assert (
        report["classes"]["railing"]["quality_state"]
        == "insufficient_segmented_views"
    )
    assert report["fused"]["classes"]["railing"]["voted_point_count"] == 0


def test_source_verified_regions_supply_object_seeds_when_models_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_clipseg(
        cloud_path: Path,
        render_dir: Path,
        output_dir: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        output_dir.mkdir(parents=True)
        return {"views": []}

    def fake_sam2(
        cloud_path: Path,
        render_dir: Path,
        clipseg_dir: Path,
        output_dir: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        output_dir.mkdir(parents=True)
        np.save(
            output_dir / "plant-votes.npy",
            np.array([0, 3, 0, 0], dtype=np.uint8),
        )
        np.save(
            output_dir / "planter-votes.npy",
            np.zeros(4, dtype=np.uint8),
        )
        return {"views": [{"status": "no-anchor"}]}

    monkeypatch.setattr(
        "plant_cleanup.scene_evidence.aggregate_clipseg_votes",
        fake_clipseg,
    )
    monkeypatch.setattr(
        "plant_cleanup.scene_evidence.aggregate_sam2_votes",
        fake_sam2,
    )
    render_dir = tmp_path / "renders"
    render_dir.mkdir()
    np.save(
        render_dir / "orbit-000-source-ids.npy",
        np.array([[1, 2], [-1, 3]], dtype=np.int64),
    )

    report = run_scene_evidence(
        tmp_path / "source.ply",
        render_dir,
        tmp_path / "scene",
        {
            "schema_version": 1,
            "scan_id": "scan",
            "plant_prompt": "complete plants",
            "classes": [
                {
                    "id": "fence",
                    "prompt": "rigid fence",
                    "distractor_prompt": "living stems",
                    "required_segmented_views": 1,
                    "manual_seed_regions": [
                        {
                            "view": "orbit-000",
                            "bounds": [0.0, 0.0, 0.5, 1.0],
                        }
                    ],
                }
            ],
        },
        clipseg_predictor=object(),
        sam2_predictor=object(),
    )

    votes = np.load(
        tmp_path / "scene" / "fused" / "fence-votes.npy"
    )
    manual = np.load(
        tmp_path
        / "scene"
        / "fused"
        / "fence-manual-seed-mask.npy"
    )
    assert votes.tolist() == [0, 0, 0, 0]
    assert manual.tolist() == [False, True, False, False]
    assert report["classes"]["fence"]["manual_seed_regions"] == {
        "region_count": 1,
        "selected_point_count": 1,
        "views": ["orbit-000"],
    }
    assert report["classes"]["fence"]["quality_state"] == "usable_manual"
