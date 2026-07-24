from __future__ import annotations

from pathlib import Path

import numpy as np

from plant_cleanup.plyio import VERTEX_DTYPE
from plant_cleanup.semantic_refine import (
    SemanticParameters,
    refine_with_semantics,
)


def test_ground_surface_evidence_completes_raised_turf_without_raised_plants(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    turf = np.array(
        [(float(x), float(y), 3.0) for x in range(3) for y in range(3)]
    )
    raised_plant = np.array([(1.0, 1.0, 4.5), (1.0, 1.0, 5.5)])
    coordinates = np.vstack((turf, raised_plant))
    cloud = np.zeros(len(coordinates), dtype=VERTEX_DTYPE)
    cloud["x"], cloud["y"], cloud["z"] = coordinates.T
    cloud["nz"][: len(turf)] = 1.0
    cloud["nx"][len(turf) :] = 1.0
    cloud["green"] = 140
    cloud["red"] = 40
    cloud["blue"] = 35
    monkeypatch.setattr(
        "plant_cleanup.semantic_refine.read_cloud",
        lambda path: cloud,
    )
    monkeypatch.setattr(
        "plant_cleanup.semantic_refine.write_decision_cloud",
        lambda *args, **kwargs: None,
    )
    geometry_decisions = np.ones(len(cloud), dtype=np.uint8)
    generic_plant = np.full(len(cloud), 3, dtype=np.uint8)
    generic_background = np.zeros(len(cloud), dtype=np.uint8)
    turf_votes = np.zeros(len(cloud), dtype=np.uint8)
    turf_votes[0] = 4

    report = refine_with_semantics(
        tmp_path / "source.ply",
        geometry_decisions,
        generic_plant,
        generic_background,
        support_height=0.0,
        support_cutoff=0.0,
        output_dir=tmp_path / "output",
        parameters=SemanticParameters(growth_radius=1.1),
        background_class_votes={"turf_ground": turf_votes},
        background_class_plant_votes={
            "turf_ground": np.zeros(len(cloud), dtype=np.uint8)
        },
        background_class_policies={"turf_ground": "ground_surface"},
    )

    decisions = np.load(tmp_path / "output" / "decision-codes.npy")
    assert np.all(decisions[: len(turf)] == 4)
    assert np.all(decisions[len(turf) :] == 1)
    assert (
        report["ground_surface_completion"]["completed_points"]
        == len(turf)
    )


def test_planned_ground_uses_paired_evidence_when_generic_model_calls_it_plant(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    turf = np.array(
        [(float(x), float(y), 3.0) for x in range(3) for y in range(3)]
    )
    raised_plant = np.array([(1.0, 1.0, 4.5), (1.0, 1.0, 5.5)])
    coordinates = np.vstack((turf, raised_plant))
    cloud = np.zeros(len(coordinates), dtype=VERTEX_DTYPE)
    cloud["x"], cloud["y"], cloud["z"] = coordinates.T
    cloud["nz"][: len(turf)] = 1.0
    cloud["nx"][len(turf) :] = 1.0
    cloud["green"] = 140
    cloud["red"] = 40
    cloud["blue"] = 35
    monkeypatch.setattr(
        "plant_cleanup.semantic_refine.read_cloud",
        lambda path: cloud,
    )
    monkeypatch.setattr(
        "plant_cleanup.semantic_refine.write_decision_cloud",
        lambda *args, **kwargs: None,
    )
    geometry_decisions = np.ones(len(cloud), dtype=np.uint8)
    generic_plant = np.full(len(cloud), 5, dtype=np.uint8)
    generic_background = np.zeros(len(cloud), dtype=np.uint8)
    turf_votes = np.zeros(len(cloud), dtype=np.uint8)
    turf_votes[0] = 2
    paired_plant_votes = np.zeros(len(cloud), dtype=np.uint8)

    report = refine_with_semantics(
        tmp_path / "source.ply",
        geometry_decisions,
        generic_plant,
        generic_background,
        support_height=0.0,
        support_cutoff=0.0,
        output_dir=tmp_path / "output",
        parameters=SemanticParameters(growth_radius=1.1),
        background_class_votes={"turf_ground": turf_votes},
        background_class_plant_votes={
            "turf_ground": paired_plant_votes
        },
        background_class_policies={"turf_ground": "ground_surface"},
    )

    decisions = np.load(tmp_path / "output" / "decision-codes.npy")
    assert np.all(decisions[: len(turf)] == 4)
    assert np.all(decisions[len(turf) :] == 1)
    assert report["scene_class_seed_points"]["turf_ground"] == 1
