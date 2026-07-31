from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

PROTECTED_CLEANED_PLY = "plant-cleaned-garden-ec2fbd1-final-v2.ply"
CORRECTED_CLEANED_PLY = (
    "plant-cleaned-garden-ec2fbd1-final-v2-orientation-corrected-v1.ply"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_cleaned_ply_from_verified_backup_atomically(
    *,
    installed: Path,
    verified_backup: Path,
    archive: Path,
    expected_installed_sha256: str,
    expected_backup_sha256: str,
) -> dict[str, Any]:
    """Restore a verified backup and archive the superseded installed file."""

    installed = installed.resolve()
    verified_backup = verified_backup.resolve()
    archive = archive.resolve()
    if installed.name != PROTECTED_CLEANED_PLY:
        raise ValueError("installed file is not the protected cleaned PLY filename")
    if (
        installed == verified_backup
        or installed == archive
        or verified_backup == archive
    ):
        raise ValueError(
            "installed file, verified backup, and archive must be distinct"
        )
    if archive.parent == installed.parent:
        raise ValueError("archive must remain outside the scan folder")
    if not installed.is_file() or not verified_backup.is_file():
        raise FileNotFoundError(
            "installed file and verified backup must both exist"
        )
    if archive.exists():
        raise FileExistsError(f"archive already exists: {archive}")

    installed_sha256 = _sha256(installed)
    if installed_sha256 != expected_installed_sha256:
        raise ValueError("installed file hash changed since recovery planning")
    backup_sha256 = _sha256(verified_backup)
    if backup_sha256 != expected_backup_sha256:
        raise ValueError("verified backup hash changed since recovery planning")

    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(installed, archive)
    archive_sha256 = _sha256(archive)
    if archive_sha256 != installed_sha256:
        raise OSError("archive verification failed")

    temporary = installed.with_name(
        f".{installed.name}.recovery-{uuid4().hex}.partial"
    )
    try:
        shutil.copy2(verified_backup, temporary)
        if _sha256(temporary) != backup_sha256:
            raise OSError("same-volume backup copy verification failed")
        os.replace(temporary, installed)
    finally:
        temporary.unlink(missing_ok=True)
    restored_sha256 = _sha256(installed)
    if restored_sha256 != backup_sha256:
        raise OSError("restored backup verification failed")
    return {
        "schema_version": 1,
        "status": "restored",
        "installed": str(installed),
        "verified_backup": str(verified_backup),
        "archive": str(archive),
        "installed_before_restore_sha256": installed_sha256,
        "archive_sha256": archive_sha256,
        "restored_sha256": restored_sha256,
        "protected_cleanup_restored": True,
    }


def write_corrected_ply_atomically(
    *,
    source: Path,
    candidate: Path,
    destination: Path,
    expected_source_sha256: str,
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    """Install a reviewed correction beside an unchanged cleanup source."""

    source = source.resolve()
    candidate = candidate.resolve()
    destination = destination.resolve()
    if source.name != PROTECTED_CLEANED_PLY:
        raise ValueError("source is not the protected cleaned PLY filename")
    if destination.name != CORRECTED_CLEANED_PLY:
        raise ValueError("destination is not the corrected PLY filename")
    if destination.parent != source.parent:
        raise ValueError("corrected output must be beside its cleanup source")
    if source == candidate or source == destination or candidate == destination:
        raise ValueError("source, candidate, and destination must be distinct")
    if not source.is_file() or not candidate.is_file():
        raise FileNotFoundError("source and candidate must both exist")
    if destination.exists():
        raise FileExistsError(
            f"corrected output already exists: {destination}"
        )

    source_sha256 = _sha256(source)
    if source_sha256 != expected_source_sha256:
        raise ValueError("source hash changed since candidate generation")
    candidate_sha256 = _sha256(candidate)
    if candidate_sha256 != expected_candidate_sha256:
        raise ValueError("candidate hash changed since verification")

    temporary = destination.with_name(
        f".{destination.name}.orientation-{uuid4().hex}.partial"
    )
    try:
        shutil.copy2(candidate, temporary)
        if _sha256(temporary) != candidate_sha256:
            raise OSError("same-volume candidate copy verification failed")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    installed_sha256 = _sha256(destination)
    if installed_sha256 != candidate_sha256:
        raise OSError("installed corrected output verification failed")
    if _sha256(source) != source_sha256:
        raise OSError("cleanup source changed while writing corrected output")
    return {
        "schema_version": 1,
        "status": "created",
        "source": str(source),
        "destination": str(destination),
        "candidate": str(candidate),
        "source_sha256": source_sha256,
        "installed_sha256": installed_sha256,
        "source_unchanged": True,
    }
