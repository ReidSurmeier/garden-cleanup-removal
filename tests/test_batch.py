from __future__ import annotations

import json
from pathlib import Path

import pytest

from railing_removal.batch import load_batch_manifest, run_batch


def test_batch_manifest_requires_unique_scan_ids_and_existing_inputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ply"
    config = tmp_path / "config.json"
    source.write_bytes(b"ply")
    config.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scans": [
                    {
                        "scan_id": "same",
                        "source": str(source),
                        "config": str(config),
                    },
                    {
                        "scan_id": "same",
                        "source": str(source),
                        "config": str(config),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_batch_manifest(manifest)


def test_batch_refuses_to_overwrite_completed_batch(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    config = tmp_path / "config.json"
    source.write_bytes(b"ply")
    config.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scans": [
                    {
                        "scan_id": "scan",
                        "source": str(source),
                        "config": str(config),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "batch-report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already complete"):
        run_batch(manifest, output)


def test_batch_flags_partial_directory_without_overwriting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.ply"
    config = tmp_path / "config.json"
    source.write_bytes(b"ply")
    config.write_text(
        json.dumps(
            {
                "vision": {
                    "clipseg_model": "clipseg",
                    "sam2_model": "sam2",
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scans": [
                    {
                        "scan_id": "partial-scan",
                        "source": str(source),
                        "config": str(config),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    partial = output / "partial-scan"
    partial.mkdir(parents=True)
    marker = partial / "preserve-me.txt"
    marker.write_text("source evidence\n", encoding="utf-8")

    monkeypatch.setattr(
        "railing_removal.batch.HuggingFaceClipSegPredictor",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "railing_removal.batch.HuggingFaceSam2Predictor",
        lambda _: object(),
    )

    report = run_batch(manifest, output)

    assert report["summary"] == {
        "complete": 0,
        "partial": 1,
        "failed": 0,
    }
    assert marker.read_text(encoding="utf-8") == "source evidence\n"
