from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from railing_removal.atomic_orientation_writeback import (
    replace_cleaned_ply_atomically,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_writeback_preserves_original_backup_and_other_files(
    tmp_path: Path,
) -> None:
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    source = scan_dir / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
    source.write_bytes(b"original-cleaned-ply")
    sentinel = scan_dir / "source-photo.png"
    sentinel.write_bytes(b"must-not-change")
    candidate = tmp_path / "candidate.ply"
    candidate.write_bytes(b"normalized-cleaned-ply")
    backup = tmp_path / "backup" / source.name

    report = replace_cleaned_ply_atomically(
        source=source,
        candidate=candidate,
        backup=backup,
        expected_source_sha256=_sha256(source),
        expected_candidate_sha256=_sha256(candidate),
    )

    assert source.read_bytes() == b"normalized-cleaned-ply"
    assert backup.read_bytes() == b"original-cleaned-ply"
    assert sentinel.read_bytes() == b"must-not-change"
    assert report["status"] == "replaced"
    assert report["backup_sha256"] != report["installed_sha256"]


def test_atomic_writeback_fails_before_writing_if_source_drifted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
    source.write_bytes(b"changed")
    candidate = tmp_path / "candidate.ply"
    candidate.write_bytes(b"normalized")
    backup = tmp_path / "backup" / source.name

    with pytest.raises(ValueError, match="source hash changed"):
        replace_cleaned_ply_atomically(
            source=source,
            candidate=candidate,
            backup=backup,
            expected_source_sha256=hashlib.sha256(b"original").hexdigest(),
            expected_candidate_sha256=_sha256(candidate),
        )

    assert source.read_bytes() == b"changed"
    assert not backup.exists()


def test_atomic_writeback_refuses_any_other_source_filename(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw-source.ply"
    source.write_bytes(b"source")
    candidate = tmp_path / "candidate.ply"
    candidate.write_bytes(b"candidate")

    with pytest.raises(ValueError, match="protected cleaned PLY filename"):
        replace_cleaned_ply_atomically(
            source=source,
            candidate=candidate,
            backup=tmp_path / "backup.ply",
            expected_source_sha256=_sha256(source),
            expected_candidate_sha256=_sha256(candidate),
        )
