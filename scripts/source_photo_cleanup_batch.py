from __future__ import annotations

import argparse
import json
from pathlib import Path

from plant_cleanup.source_photo_cleanup_batch import (
    run_source_photo_cleanup_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build publishable clean clouds from calibrated photo evidence."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("source_photo_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    report = run_source_photo_cleanup_batch(
        args.manifest,
        args.baseline_root,
        args.source_photo_root,
        args.output_root,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
