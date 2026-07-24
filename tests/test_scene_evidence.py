from __future__ import annotations

from pathlib import Path

import numpy as np

from plant_cleanup.scene_evidence import fuse_scene_votes


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
