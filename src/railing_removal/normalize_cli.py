from __future__ import annotations

import argparse
import json
from pathlib import Path

from railing_removal.normalize_run import normalize_cleanup_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a frozen cleanup run from its ground decisions and "
            "read-only Metashape camera inventory."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("cleanup_run", type=Path)
    parser.add_argument("camera_inventory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    report = normalize_cleanup_run(
        args.source,
        args.cleanup_run,
        args.camera_inventory,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": report["plan"]["status"],
                "scale": report["plan"]["scale"],
                "ground_point_count": report["ground_evidence"][
                    "point_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
