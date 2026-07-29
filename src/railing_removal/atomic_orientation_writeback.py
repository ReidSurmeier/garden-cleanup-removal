from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
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


def replace_cleaned_ply_atomically(
    *,
    source: Path,
    candidate: Path,
    backup: Path,
    expected_source_sha256: str,
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    """Install one verified orientation while retaining an external backup."""

    source = source.resolve()
    candidate = candidate.resolve()
    backup = backup.resolve()
    if source.name != PROTECTED_CLEANED_PLY:
        raise ValueError("source is not the protected cleaned PLY filename")
    if source == candidate or source == backup or candidate == backup:
        raise ValueError("source, candidate, and backup must be distinct")
    if backup.parent == source.parent:
        raise ValueError("backup must remain outside the scan folder")
    if not source.is_file() or not candidate.is_file():
        raise FileNotFoundError("source and candidate must both exist")
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")

    source_sha256 = _sha256(source)
    if source_sha256 != expected_source_sha256:
        raise ValueError("source hash changed since candidate generation")
    candidate_sha256 = _sha256(candidate)
    if candidate_sha256 != expected_candidate_sha256:
        raise ValueError("candidate hash changed since verification")

    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, backup)
    backup_sha256 = _sha256(backup)
    if backup_sha256 != source_sha256:
        raise OSError("backup verification failed")

    temporary = source.with_name(
        f".{source.name}.orientation-{uuid4().hex}.partial"
    )
    try:
        shutil.copy2(candidate, temporary)
        if _sha256(temporary) != candidate_sha256:
            raise OSError("same-volume candidate copy verification failed")
        os.replace(temporary, source)
    finally:
        temporary.unlink(missing_ok=True)
    installed_sha256 = _sha256(source)
    if installed_sha256 != candidate_sha256:
        raise OSError("installed candidate verification failed")
    return {
        "schema_version": 1,
        "status": "replaced",
        "source": str(source),
        "backup": str(backup),
        "candidate": str(candidate),
        "original_sha256": source_sha256,
        "backup_sha256": backup_sha256,
        "installed_sha256": installed_sha256,
        "only_protected_cleaned_ply_replaced": True,
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
