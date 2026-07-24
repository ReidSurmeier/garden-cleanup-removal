from __future__ import annotations

import json
from pathlib import Path

from plant_cleanup.source_photo_batch import (
    run_source_photo_semantic_batch,
)


def _manifest(path: Path, root: Path) -> None:
    projects = []
    for scan_id in ("scan-1", "scan-2"):
        project = root / f"{scan_id}.psx"
        project.write_text("source project", encoding="utf-8")
        photo_dir = root / f"{scan_id}-photos"
        photo_dir.mkdir()
        projects.append(
            {
                "scan_id": scan_id,
                "project": str(project),
                "photo_dir": str(photo_dir),
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": projects,
            }
        ),
        encoding="utf-8",
    )


def _dependencies(root: Path) -> tuple[Path, Path, Path]:
    canonical_root = root / "canonical"
    native_root = root / "native"
    inventory_root = root / "inventory"
    for scan_id in ("scan-1", "scan-2"):
        canonical = canonical_root / scan_id
        native = native_root / scan_id
        inventory = inventory_root / scan_id
        canonical.mkdir(parents=True)
        native.mkdir(parents=True)
        inventory.mkdir(parents=True)
        (canonical / "source-stride8-zup.ply").write_bytes(b"canonical")
        (native / "source-native.ply").write_bytes(b"native")
        (inventory / "camera-inventory.json").write_text(
            json.dumps({"project_opened_read_only": True}),
            encoding="utf-8",
        )
    return canonical_root, native_root, inventory_root


def test_source_photo_batch_caches_predictor_and_resumes(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "projects.json"
    _manifest(manifest, tmp_path)
    canonical, native, inventories = _dependencies(tmp_path)
    output = tmp_path / "semantic"
    complete = output / "scan-1"
    complete.mkdir(parents=True)
    (complete / "report.json").write_text(
        json.dumps(
            {
                "project_opened_read_only": True,
                "point_count": 100,
                "unseen_point_count": 2,
            }
        ),
        encoding="utf-8",
    )
    predictor = object()
    calls: list[tuple[Path, Path, Path, Path, object]] = []

    def aggregate(
        canonical_cloud: Path,
        native_cloud: Path,
        inventory: Path,
        photo_root: Path,
        destination: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append(
            (
                canonical_cloud,
                native_cloud,
                inventory,
                photo_root,
                kwargs["predictor"],
            )
        )
        destination.mkdir()
        report = {
            "project_opened_read_only": True,
            "point_count": 200,
            "unseen_point_count": 3,
        }
        (destination / "report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return report

    report = run_source_photo_semantic_batch(
        manifest,
        canonical,
        native,
        inventories,
        output,
        predictor=predictor,
        model_id="model",
        plant_labels=("plant",),
        stride=8,
        camera_count=12,
        maximum_dimension=768,
        aggregate=aggregate,
    )

    assert len(calls) == 1
    assert calls[0][0] == (
        canonical / "scan-2" / "source-stride8-zup.ply"
    ).resolve()
    assert calls[0][-1] is predictor
    assert report["summary"] == {
        "complete": 2,
        "partial": 0,
        "failed": 0,
    }


def test_source_photo_batch_preserves_partial_output(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "projects.json"
    _manifest(manifest, tmp_path)
    canonical, native, inventories = _dependencies(tmp_path)
    output = tmp_path / "semantic"
    for scan_id in ("scan-1", "scan-2"):
        partial = output / scan_id
        partial.mkdir(parents=True)
        (partial / "evidence.txt").write_text(
            "preserve",
            encoding="utf-8",
        )

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("partial directories must not be processed")

    report = run_source_photo_semantic_batch(
        manifest,
        canonical,
        native,
        inventories,
        output,
        predictor=object(),
        model_id="model",
        plant_labels=("plant",),
        aggregate=forbidden,
    )

    assert report["summary"] == {
        "complete": 0,
        "partial": 2,
        "failed": 0,
    }
    assert (
        output / "scan-1" / "evidence.txt"
    ).read_text(encoding="utf-8") == "preserve"
