from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from plant_cleanup.cloud_render import render_cloud_views  # noqa: E402
from railing_removal.normalization import write_normalized_cloud  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build derived before/after orientation review artifacts."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite orientation review: {args.output_root}"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    plans = {item["scan_id"]: item for item in analysis["scans"]}
    reports: list[dict[str, object]] = []
    for item in manifest["scans"]:
        scan_id = item["scan_id"]
        plan = plans[scan_id]
        if "matrix" not in plan:
            reports.append(
                {
                    "scan_id": scan_id,
                    "status": plan["status"],
                    "error": plan.get("error", "normalization matrix missing"),
                }
            )
            continue
        scan_dir = args.output_root / scan_id.replace(" ", "-").replace(":", "")
        normalized_dir = scan_dir / "normalized"
        normalized_dir.mkdir(parents=True)
        plant = Path(item["plant"])
        rejected = Path(item["rejected"])
        normalized_plant = normalized_dir / "plant-normalized.ply"
        normalized_rejected = normalized_dir / "rejected-normalized.ply"
        layer_reports = {
            "plant": write_normalized_cloud(
                plant,
                normalized_plant,
                plan["matrix"],
            ),
            "rejected": write_normalized_cloud(
                rejected,
                normalized_rejected,
                plan["matrix"],
            ),
        }
        render_reports = {
            "before": render_cloud_views(
                plant,
                scan_dir / "before-plant",
                size=1600,
                point_radius=1,
            ),
            "after": render_cloud_views(
                normalized_plant,
                scan_dir / "after-plant",
                size=1600,
                point_radius=1,
            ),
            "rejected": render_cloud_views(
                normalized_rejected,
                scan_dir / "after-rejected",
                size=1600,
                point_radius=1,
            ),
        }
        report = {
            "schema_version": 1,
            "scan_id": scan_id,
            "status": plan["status"],
            "scale": plan["scale"],
            "matrix": plan["matrix"],
            "layers": layer_reports,
            "renders": render_reports,
        }
        report_path = scan_dir / "orientation-review-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports.append({**report, "report": str(report_path)})
    batch_report = {
        "schema_version": 1,
        "scans": reports,
    }
    batch_path = args.output_root / "orientation-review-batch.json"
    batch_path.write_text(
        json.dumps(batch_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_root),
                "scan_count": len(reports),
                "statuses": {
                    status: sum(
                        report["status"] == status for report in reports
                    )
                    for status in sorted(
                        {str(report["status"]) for report in reports}
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
