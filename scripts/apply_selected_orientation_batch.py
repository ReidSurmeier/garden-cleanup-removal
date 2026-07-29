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
    replace_cleaned_ply_atomically,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: apply_selected_orientation_batch.py "
            "PROJECTS.json SELECTED_REVIEW_ROOT BACKUP_ROOT "
            "WRITEBACK_REPORT.json"
        )
    manifest_path = Path(sys.argv[1]).resolve()
    review_root = Path(sys.argv[2]).resolve()
    backup_root = Path(sys.argv[3]).resolve()
    report_path = Path(sys.argv[4]).resolve()
    if report_path.exists():
        raise FileExistsError(f"writeback report already exists: {report_path}")
    manifest = _read_json(manifest_path)
    planned: list[dict[str, Any]] = []
    for item in manifest["projects"]:
        scan_id = str(item["scan_id"])
        scan_dir = Path(item["project"]).resolve().parent
        source = scan_dir / PROTECTED_CLEANED_PLY
        review = _read_json(review_root / scan_id / "review-report.json")
        layer = review["candidate_layer"]
        candidate = Path(layer["normalized"]).resolve()
        backup = backup_root / scan_id / PROTECTED_CLEANED_PLY
        if Path(review["source_cleaned_ply"]).resolve() != source:
            raise ValueError(f"{scan_id} review source path mismatch")
        if not candidate.is_relative_to(review_root):
            raise ValueError(f"{scan_id} candidate is outside review root")
        if int(layer["source_point_count"]) != int(
            layer["normalized_point_count"]
        ):
            raise ValueError(f"{scan_id} candidate point count changed")
        if not bool(layer["source_identity_preserved"]):
            raise ValueError(f"{scan_id} source identity was not preserved")
        if _sha256(source) != layer["source_sha256"]:
            raise ValueError(f"{scan_id} source changed before writeback")
        if _sha256(candidate) != layer["normalized_sha256"]:
            raise ValueError(f"{scan_id} candidate changed before writeback")
        if backup.exists():
            raise FileExistsError(f"{scan_id} backup already exists")
        planned.append(
            {
                "scan_id": scan_id,
                "source": str(source),
                "candidate": str(candidate),
                "backup": str(backup),
                "expected_source_sha256": layer["source_sha256"],
                "expected_candidate_sha256": layer["normalized_sha256"],
                "point_count": int(layer["source_point_count"]),
                "non_target_files_before": _non_target_file_hashes(scan_dir),
                "status": "preflight_passed",
            }
        )
    journal = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "review_root": str(review_root),
        "backup_root": str(backup_root),
        "preflight_complete_before_mutation": True,
        "allowed_source_filename": PROTECTED_CLEANED_PLY,
        "results": planned,
    }
    _write_journal(report_path, journal)
    for item in journal["results"]:
        result = replace_cleaned_ply_atomically(
            source=Path(item["source"]),
            candidate=Path(item["candidate"]),
            backup=Path(item["backup"]),
            expected_source_sha256=item["expected_source_sha256"],
            expected_candidate_sha256=item["expected_candidate_sha256"],
        )
        item.update(result)
        if _non_target_file_hashes(Path(item["source"]).parent) != item[
            "non_target_files_before"
        ]:
            raise OSError(
                f"{item['scan_id']} non-target scan-folder file changed"
            )
        item["non_target_files_unchanged"] = True
        _write_journal(report_path, journal)
        print(f"replaced {item['scan_id']}", flush=True)
    journal["status"] = "complete"
    journal["replaced_count"] = len(journal["results"])
    _write_journal(report_path, journal)
    print(json.dumps(journal, indent=2), flush=True)


if __name__ == "__main__":
    main()
