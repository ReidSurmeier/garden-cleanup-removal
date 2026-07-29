from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.atomic_orientation_writeback import (  # noqa: E402
    PROTECTED_CLEANED_PLY,
)
from railing_removal.orientation_selection import (  # noqa: E402
    orientation_review_candidates,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(
            "usage: build_blender_orientation_manifest.py "
            "PROJECTS.json FUSED_EVIDENCE_ROOT OUTPUT.json SCAN_ID..."
        )
    projects_path = Path(sys.argv[1]).resolve()
    fused_root = Path(sys.argv[2]).resolve()
    output = Path(sys.argv[3]).resolve()
    requested = list(dict.fromkeys(sys.argv[4:]))
    if output.exists():
        raise FileExistsError(f"manifest already exists: {output}")
    projects = {
        str(item["scan_id"]): item
        for item in _read_json(projects_path)["projects"]
    }
    scans: list[dict[str, Any]] = []
    for scan_id in requested:
        item = projects.get(scan_id)
        if item is None:
            raise ValueError(f"unknown scan id: {scan_id}")
        source = Path(item["project"]).resolve().parent / PROTECTED_CLEANED_PLY
        if not source.is_file():
            raise FileNotFoundError(source)
        fused_path = fused_root / scan_id / "fused-orientation.json"
        fused = _read_json(fused_path)
        scans.append(
            {
                "scan_id": scan_id,
                "source": str(source),
                "source_opened_read_only": True,
                "fused_evidence": str(fused_path),
                "candidates": orientation_review_candidates(fused),
            }
        )
    manifest = {
        "schema_version": 1,
        "projects_manifest": str(projects_path),
        "source_files_opened_read_only": True,
        "scans": scans,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
