from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from railing_removal.atomic_orientation_writeback import (
    CORRECTED_CLEANED_PLY,
    restore_cleaned_ply_from_verified_backup_atomically,
    write_corrected_ply_atomically,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovery_restores_verified_backup_and_preserves_other_files(
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

    report = restore_cleaned_ply_from_verified_backup_atomically(
        installed=source,
        verified_backup=candidate,
        archive=backup,
        expected_installed_sha256=_sha256(source),
        expected_backup_sha256=_sha256(candidate),
    )

    assert source.read_bytes() == b"normalized-cleaned-ply"
    assert backup.read_bytes() == b"original-cleaned-ply"
    assert sentinel.read_bytes() == b"must-not-change"
    assert report["status"] == "restored"
    assert report["protected_cleanup_restored"] is True
    assert report["archive_sha256"] != report["restored_sha256"]


def test_recovery_fails_before_writing_if_installed_file_drifted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
    source.write_bytes(b"changed")
    candidate = tmp_path / "candidate.ply"
    candidate.write_bytes(b"normalized")
    backup = tmp_path / "backup" / source.name

    with pytest.raises(ValueError, match="installed file hash changed"):
        restore_cleaned_ply_from_verified_backup_atomically(
            installed=source,
            verified_backup=candidate,
            archive=backup,
            expected_installed_sha256=hashlib.sha256(b"original").hexdigest(),
            expected_backup_sha256=_sha256(candidate),
        )

    assert source.read_bytes() == b"changed"
    assert not backup.exists()


def test_recovery_refuses_any_other_source_filename(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw-source.ply"
    source.write_bytes(b"source")
    candidate = tmp_path / "candidate.ply"
    candidate.write_bytes(b"candidate")

    with pytest.raises(ValueError, match="protected cleaned PLY filename"):
        restore_cleaned_ply_from_verified_backup_atomically(
            installed=source,
            verified_backup=candidate,
            archive=tmp_path / "backup.ply",
            expected_installed_sha256=_sha256(source),
            expected_backup_sha256=_sha256(candidate),
        )


def test_corrected_output_is_created_without_changing_cleanup_source(
    tmp_path: Path,
) -> None:
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    source = scan_dir / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
    source.write_bytes(b"source-cleaned-ply")
    candidate = tmp_path / "reviewed-candidate.ply"
    candidate.write_bytes(b"orientation-corrected-ply")
    sentinel = scan_dir / "source-photo.png"
    sentinel.write_bytes(b"must-not-change")
    destination = scan_dir / CORRECTED_CLEANED_PLY

    report = write_corrected_ply_atomically(
        source=source,
        candidate=candidate,
        destination=destination,
        expected_source_sha256=_sha256(source),
        expected_candidate_sha256=_sha256(candidate),
    )

    assert source.read_bytes() == b"source-cleaned-ply"
    assert destination.read_bytes() == b"orientation-corrected-ply"
    assert sentinel.read_bytes() == b"must-not-change"
    assert report["source_unchanged"] is True
    assert report["installed_sha256"] == _sha256(candidate)


def test_corrected_output_refuses_to_overwrite_an_existing_result(
    tmp_path: Path,
) -> None:
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    source = scan_dir / "plant-cleaned-garden-ec2fbd1-final-v2.ply"
    source.write_bytes(b"source")
    candidate = tmp_path / "candidate.ply"
    candidate.write_bytes(b"candidate")
    destination = scan_dir / CORRECTED_CLEANED_PLY
    destination.write_bytes(b"existing-corrected-output")

    with pytest.raises(FileExistsError, match="already exists"):
        write_corrected_ply_atomically(
            source=source,
            candidate=candidate,
            destination=destination,
            expected_source_sha256=_sha256(source),
            expected_candidate_sha256=_sha256(candidate),
        )

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"existing-corrected-output"
