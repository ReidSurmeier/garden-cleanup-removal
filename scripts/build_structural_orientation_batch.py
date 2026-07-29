from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.structural_orientation import (  # noqa: E402
    build_structural_orientation_report,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8-sig"))


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: build_structural_orientation_batch.py "
            "PRODUCTION_PLAN.json CAMERA_INVENTORY_ROOT OUTPUT_ROOT"
        )
    plan_path = Path(sys.argv[1]).resolve()
    inventory_root = Path(sys.argv[2]).resolve()
    output_root = Path(sys.argv[3]).resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)
    plan = _read(plan_path)
    results = []
    for index, item in enumerate(plan["scans"], start=1):
        scan_id = str(item["scan_id"])
        print(f"[{index}/{len(plan['scans'])}] {scan_id}", flush=True)
        scan_dir = Path(item["source"]).resolve().parent
        reports = sorted(
            scan_dir.glob("plant-cleanup-report-garden-*-final-v2.json")
        )
        if len(reports) != 1:
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "failed",
                    "error": "requires exactly one final-v2 cleanup report",
                }
            )
            continue
        try:
            report = build_structural_orientation_report(
                scan_id,
                _read(
                    inventory_root / scan_id / "camera-inventory.json"
                ),
                _read(reports[0]),
            )
            destination = output_root / scan_id / "fused-orientation.json"
            _write_new(destination, report)
        except Exception as error:
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        results.append(
            {
                "scan_id": scan_id,
                "status": "complete",
                "consensus_status": report["consensus"]["status"],
                "supporting_families": report["consensus"][
                    "supporting_families"
                ],
            }
        )
    batch = {
        "schema_version": 1,
        "production_plan": str(plan_path),
        "source_data_opened_read_only": True,
        "summary": {
            status: sum(item["status"] == status for item in results)
            for status in ("complete", "failed")
        },
        "results": results,
    }
    _write_new(output_root / "batch-report.json", batch)
    print(json.dumps(batch["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
