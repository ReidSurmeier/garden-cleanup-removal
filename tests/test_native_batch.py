from __future__ import annotations

import json
from pathlib import Path

from railing_removal.native_batch import (
    run_canonicalize_native_batch,
    run_native_export_batch,
)


def _manifest(path: Path, source: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {"scan_id": "scan-a", "project": str(source)}
                ],
            }
        ),
        encoding="utf-8",
    )


def test_native_and_canonical_batches_preserve_stages_and_manifests(
    tmp_path: Path,
) -> None:
    project = tmp_path / "scan.psx"
    project.write_text("source", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, project)
    runtime = object()

    def native_export(
        project_path: Path,
        output: Path,
        metashape: object,
    ) -> dict[str, object]:
        assert metashape is runtime
        output.write_bytes(b"native")
        return {
            "output": str(output),
            "source_point_count": 10,
            "coordinate_frame": {
                "right": [1, 0, 0],
                "forward": [0, 1, 0],
                "up": [0, 0, 1],
                "source": "test",
            },
        }

    native_root = tmp_path / "native"
    native_report = run_native_export_batch(
        manifest,
        native_root,
        runtime,
        exporter=native_export,
    )
    assert native_report["summary"]["complete"] == 1

    calls: list[tuple[Path, Path, int]] = []

    def canonicalize(
        source: Path,
        output: Path,
        frame: dict[str, object],
        *,
        stride: int,
    ) -> dict[str, object]:
        calls.append((source, output, stride))
        output.write_bytes(b"canonical")
        return {
            "output": str(output),
            "exported_point_count": 5,
            "coordinate_frame": frame,
        }

    canonical_root = tmp_path / "canonical"
    canonical_report = run_canonicalize_native_batch(
        manifest,
        native_root,
        canonical_root,
        stride=2,
        canonicalizer=canonicalize,
    )

    assert canonical_report["summary"]["complete"] == 1
    assert calls[0][0] == native_root / "scan-a" / "source-native.ply"
    assert calls[0][2] == 2
    assert (
        canonical_root / "scan-a" / "source-stride2-zup.ply"
    ).read_bytes() == b"canonical"
    assert project.read_text(encoding="utf-8") == "source"


def test_native_batch_flags_partial_directory_without_overwriting(
    tmp_path: Path,
) -> None:
    project = tmp_path / "scan.psx"
    project.write_text("source", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, project)
    output_root = tmp_path / "native"
    partial_dir = output_root / "scan-a"
    partial_dir.mkdir(parents=True)
    partial = partial_dir / "source-native.ply.partial-old"
    partial.write_bytes(b"evidence")

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("partial directory must not be overwritten")

    report = run_native_export_batch(
        manifest,
        output_root,
        object(),
        exporter=forbidden,
    )

    assert report["summary"] == {
        "complete": 0,
        "partial": 1,
        "failed": 0,
    }
    assert partial.read_bytes() == b"evidence"
