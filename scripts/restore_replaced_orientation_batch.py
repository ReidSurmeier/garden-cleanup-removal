from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.atomic_orientation_writeback import (  # noqa: E402
    PROTECTED_CLEANED_PLY,
    restore_cleaned_ply_from_verified_backup_atomically,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _non_target_file_hashes(scan_dir: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(scan_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != PROTECTED_CLEANED_PLY
    }


def _write_journal(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: restore_replaced_orientation_batch.py "
            "WRITEBACK_REPORT.json REPLACED_ARCHIVE_ROOT "
            "RESTORE_REPORT.json"
        )
    writeback_path = Path(sys.argv[1]).resolve()
    archive_root = Path(sys.argv[2]).resolve()
    restore_path = Path(sys.argv[3]).resolve()
    if restore_path.exists():
        raise FileExistsError(f"restore report already exists: {restore_path}")
    writeback = json.loads(
        writeback_path.read_text(encoding="utf-8-sig")
    )
    if writeback.get("status") != "complete":
        raise ValueError("writeback report is not complete")

    planned: list[dict[str, Any]] = []
    for item in writeback["results"]:
        scan_id = str(item["scan_id"])
        source = Path(item["source"]).resolve()
        original = Path(item["backup"]).resolve()
        archived_replacement = (
            archive_root / scan_id / PROTECTED_CLEANED_PLY
        )
        if source.name != PROTECTED_CLEANED_PLY:
            raise ValueError(f"{scan_id} source filename is not protected")
        if _sha256(source) != item["installed_sha256"]:
            raise ValueError(f"{scan_id} installed file changed")
        if _sha256(original) != item["original_sha256"]:
            raise ValueError(f"{scan_id} original backup changed")
        if archived_replacement.exists():
            raise FileExistsError(
                f"{scan_id} replacement archive already exists"
            )
        planned.append(
            {
                "scan_id": scan_id,
                "source": str(source),
                "original": str(original),
                "archived_replacement": str(archived_replacement),
                "installed_sha256": item["installed_sha256"],
                "original_sha256": item["original_sha256"],
                "non_target_files_before": _non_target_file_hashes(
                    source.parent
                ),
                "status": "preflight_passed",
            }
        )

    journal: dict[str, Any] = {
        "schema_version": 1,
        "writeback_report": str(writeback_path),
        "preflight_complete_before_mutation": True,
        "results": planned,
    }
    _write_journal(restore_path, journal)
    for item in journal["results"]:
        result = restore_cleaned_ply_from_verified_backup_atomically(
            installed=Path(item["source"]),
            verified_backup=Path(item["original"]),
            archive=Path(item["archived_replacement"]),
            expected_installed_sha256=item["installed_sha256"],
            expected_backup_sha256=item["original_sha256"],
        )
        if _sha256(Path(item["source"])) != item["original_sha256"]:
            raise OSError(f"{item['scan_id']} original was not restored")
        if _non_target_file_hashes(Path(item["source"]).parent) != item[
            "non_target_files_before"
        ]:
            raise OSError(
                f"{item['scan_id']} non-target scan-folder file changed"
            )
        item.update(result)
        item["status"] = "restored"
        item["non_target_files_unchanged"] = True
        _write_journal(restore_path, journal)
        print(f"restored {item['scan_id']}", flush=True)
    journal["status"] = "complete"
    journal["restored_count"] = len(journal["results"])
    _write_journal(restore_path, journal)
    print(json.dumps(journal, indent=2), flush=True)


if __name__ == "__main__":
    main()
