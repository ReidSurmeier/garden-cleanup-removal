from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.correction_manifest import (  # noqa: E402
    build_correction_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an additive retry manifest for every unfinished baseline "
            "scan while omitting fragile legacy scene plans."
        )
    )
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = build_correction_manifest(
        args.source_manifest,
        args.baseline_report,
        args.output,
    )
    print(json.dumps({"corrections": len(manifest["scans"])}, indent=2))


if __name__ == "__main__":
    main()
