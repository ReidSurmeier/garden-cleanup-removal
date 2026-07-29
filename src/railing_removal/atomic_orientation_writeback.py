from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4


PROTECTED_CLEANED_PLY = "plant-cleaned-garden-ec2fbd1-final-v2.ply"


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
