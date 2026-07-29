from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.orientation_selection import (  # noqa: E402
    orientation_review_candidates,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8-sig"))


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: build_blender_production_review_manifest.py "
            "PRODUCTION_PLAN.json FUSED_ROOT OUTPUT.json"
        )
    plan_path = Path(sys.argv[1]).resolve()
    fused_root = Path(sys.argv[2]).resolve()
    output = Path(sys.argv[3]).resolve()
    if output.exists():
        raise FileExistsError(f"manifest already exists: {output}")
    plan = _read(plan_path)
    scans = []
    for item in plan["scans"]:
        scan_id = str(item["scan_id"])
        fused_path = fused_root / scan_id / "fused-orientation.json"
        if not fused_path.is_file():
            raise FileNotFoundError(fused_path)
        scans.append(
            {
                **item,
                "fused_evidence": str(fused_path),
                "candidates": orientation_review_candidates(
                    _read(fused_path)
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "production_plan": str(plan_path),
        "source_files_opened_read_only": True,
        "scans": scans,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(f"planned {len(scans)} Blender reviews", flush=True)


if __name__ == "__main__":
    main()
