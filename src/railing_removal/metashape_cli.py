from __future__ import annotations

import argparse
import json
from pathlib import Path

from railing_removal.metashape_reader import export_reader_cloud_readonly


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Metashape point cloud through its read-only reader."
    )
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(
            f"refusing to overwrite report: {args.report.resolve()}"
        )

    import Metashape  # type: ignore[import-not-found]

    report = export_reader_cloud_readonly(
        args.project,
        args.output,
        Metashape,
        stride=args.stride,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
