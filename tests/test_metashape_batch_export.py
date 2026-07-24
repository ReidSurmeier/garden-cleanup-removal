from __future__ import annotations

import json
from pathlib import Path

from railing_removal.metashape_batch_export import run_metashape_export_batch


def _write_manifest(path: Path, projects: list[tuple[str, Path]]) -> None:
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


def test_batch_exports_projects_and_reuses_one_runtime(tmp_path: Path) -> None:
    projects = []
    for scan_id in ("scan-a", "scan-b"):
        project = tmp_path / f"{scan_id}.psx"
        project.write_text("source", encoding="utf-8")
        projects.append((scan_id, project))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, projects)
    calls: list[tuple[Path, Path, object, int]] = []
    runtime = object()

    def export(
        project: Path,
        output: Path,
        metashape: object,
        *,
        stride: int,
    ) -> dict[str, object]:
        calls.append((project, output, metashape, stride))
        output.write_bytes(b"ply")
        return {
            "project": str(project),
            "output": str(output),
            "read_only": True,
            "exported_point_count": 3,
        }

    report = run_metashape_export_batch(
        manifest,
        tmp_path / "exports",
        runtime,
        stride=8,
        exporter=export,
    )

    assert report["summary"] == {
        "complete": 2,
        "partial": 0,
        "failed": 0,
    }
    assert len(calls) == 2
    assert all(call[2] is runtime and call[3] == 8 for call in calls)
    assert projects[0][0] in str(calls[0][1])
    assert projects[0][1].read_text(encoding="utf-8") == "source"


def test_batch_resumes_complete_and_flags_partial_without_overwrite(
    tmp_path: Path,
) -> None:
    complete_source = tmp_path / "complete.psx"
    partial_source = tmp_path / "partial.psx"
    complete_source.write_text("complete source", encoding="utf-8")
    partial_source.write_text("partial source", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            ("complete", complete_source),
            ("partial", partial_source),
        ],
    )
    output_root = tmp_path / "exports"
    complete_dir = output_root / "complete"
    complete_dir.mkdir(parents=True)
    complete_output = complete_dir / "source-stride8-zup.ply"
    complete_output.write_bytes(b"completed")
    (complete_dir / "export-report.json").write_text(
        json.dumps({"exported_point_count": 9}),
        encoding="utf-8",
    )
    partial_dir = output_root / "partial"
    partial_dir.mkdir()
    partial_file = partial_dir / "source-stride8-zup.ply.partial-old"
    partial_file.write_bytes(b"preserve partial")

    def forbidden_export(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("completed and partial scans must not be exported")

    report = run_metashape_export_batch(
        manifest,
        output_root,
        object(),
        stride=8,
        exporter=forbidden_export,
    )

    assert report["summary"] == {
        "complete": 1,
        "partial": 1,
        "failed": 0,
    }
    assert complete_output.read_bytes() == b"completed"
    assert partial_file.read_bytes() == b"preserve partial"
