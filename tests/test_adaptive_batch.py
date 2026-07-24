from __future__ import annotations

import json
from pathlib import Path

import pytest

from railing_removal.adaptive_batch import build_adaptive_batch


def _project_manifest(path: Path, projects: list[tuple[str, Path]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {"scan_id": scan_id, "project": str(project)}
                    for scan_id, project in projects
                ],
            }
        ),
        encoding="utf-8",
    )


def test_adaptive_batch_resumes_configs_and_builds_cleanup_manifest(
    tmp_path: Path,
) -> None:
    project_a = tmp_path / "a.psx"
    project_b = tmp_path / "b.psx"
    project_a.write_text("source-a", encoding="utf-8")
    project_b.write_text("source-b", encoding="utf-8")
    projects = tmp_path / "projects.json"
    _project_manifest(
        projects,
        [("scan-a", project_a), ("scan-b", project_b)],
    )
    source_root = tmp_path / "canonical"
    for scan_id in ("scan-a", "scan-b"):
        destination = source_root / scan_id
        destination.mkdir(parents=True)
        (destination / "source-stride8-zup.ply").write_bytes(
            scan_id.encode("ascii")
        )
    base_config = tmp_path / "base.json"
    base_config.write_text('{"base": true}\n', encoding="utf-8")
    output = tmp_path / "profiles"
    configs = output / "configs"
    configs.mkdir(parents=True)
    preserved = configs / "scan-a.json"
    preserved.write_text('{"preserved": true}\n', encoding="utf-8")
    calls: list[str] = []

    def builder(source: Path, base: dict[str, object]) -> dict[str, object]:
        calls.append(source.parent.name)
        return {"source": source.parent.name, "base": base}

    report = build_adaptive_batch(
        projects,
        source_root,
        output,
        base_config,
        stride=8,
        config_builder=builder,
    )

    assert calls == ["scan-b"]
    assert json.loads(preserved.read_text(encoding="utf-8")) == {
        "preserved": True
    }
    manifest = json.loads(
        (output / "cleanup-manifest.json").read_text(encoding="utf-8")
    )
    assert [item["scan_id"] for item in manifest["scans"]] == [
        "scan-a",
        "scan-b",
    ]
    assert report["summary"] == {"complete": 2}
    assert project_a.read_text(encoding="utf-8") == "source-a"

    with pytest.raises(FileExistsError, match="already finalized"):
        build_adaptive_batch(
            projects,
            source_root,
            output,
            base_config,
            stride=8,
            config_builder=builder,
        )


def test_adaptive_batch_leaves_existing_partial_source_unmodified(
    tmp_path: Path,
) -> None:
    project = tmp_path / "a.psx"
    project.write_text("source", encoding="utf-8")
    projects = tmp_path / "projects.json"
    _project_manifest(projects, [("scan-a", project)])
    source_root = tmp_path / "canonical"
    partial = source_root / "scan-a"
    partial.mkdir(parents=True)
    marker = partial / "source-stride8-zup.ply.partial-old"
    marker.write_bytes(b"partial evidence")
    base_config = tmp_path / "base.json"
    base_config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="canonical source"):
        build_adaptive_batch(
            projects,
            source_root,
            tmp_path / "profiles",
            base_config,
            stride=8,
            config_builder=lambda *_: {},
        )

    assert marker.read_bytes() == b"partial evidence"


def test_adaptive_batch_materializes_scan_bound_scene_plans(
    tmp_path: Path,
) -> None:
    project = tmp_path / "a.psx"
    project.write_text("source", encoding="utf-8")
    projects = tmp_path / "projects.json"
    _project_manifest(projects, [("scan-a", project)])
    canonical = tmp_path / "canonical" / "scan-a"
    canonical.mkdir(parents=True)
    (canonical / "source-stride8-zup.ply").write_bytes(b"cloud")
    base_config = tmp_path / "base.json"
    base_config.write_text("{}\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plant_prompt": "complete plants including roots",
                "classes": {
                    "railing": {
                        "prompt": "rigid metal railing",
                        "distractor_prompt": "branches roots and stems",
                        "anchor_strategy": "semantic",
                        "decision_policy": "class_exclusive_background",
                        "background_depth_fraction": 0.006,
                        "background_anchor_limit": 12,
                        "required_segmented_views": 1,
                    }
                },
                "assignments": {"scan-a": ["railing"]},
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "profiles"
    build_adaptive_batch(
        projects,
        tmp_path / "canonical",
        output,
        base_config,
        stride=8,
        scene_catalog_path=catalog,
        config_builder=lambda *_: {},
    )

    manifest = json.loads(
        (output / "cleanup-manifest.json").read_text(encoding="utf-8")
    )
    plan_path = Path(manifest["scans"][0]["scene_plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["scan_id"] == "scan-a"
    assert [value["id"] for value in plan["classes"]] == ["railing"]


def test_adaptive_batch_rejects_catalog_assignments_outside_manifest(
    tmp_path: Path,
) -> None:
    project = tmp_path / "a.psx"
    project.write_text("source", encoding="utf-8")
    projects = tmp_path / "projects.json"
    _project_manifest(projects, [("scan-a", project)])
    canonical = tmp_path / "canonical" / "scan-a"
    canonical.mkdir(parents=True)
    (canonical / "source-stride8-zup.ply").write_bytes(b"cloud")
    base_config = tmp_path / "base.json"
    base_config.write_text("{}\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plant_prompt": "plants",
                "classes": {"railing": {"prompt": "railing"}},
                "assignments": {"not-a-scan": ["railing"]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown scan"):
        build_adaptive_batch(
            projects,
            tmp_path / "canonical",
            tmp_path / "profiles",
            base_config,
            stride=8,
            scene_catalog_path=catalog,
            config_builder=lambda *_: {},
        )
