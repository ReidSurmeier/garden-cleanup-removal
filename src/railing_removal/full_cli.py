from __future__ import annotations

import argparse
import json
from pathlib import Path

from railing_removal.full_pipeline import run_full_cleanup


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete plant cleanup model: geometry, CLIPSeg, SAM2, "
            "uncertain-floor removal, railing evidence, and rail completion."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--railing-plan",
        type=Path,
        help="Enable railing removal using this scene plan.",
    )
    args = parser.parse_args()
    report = run_full_cleanup(
        args.source,
        args.output,
        config_path=args.config,
        railing_plan_path=args.railing_plan,
        progress=lambda stage: print(f"[garden-full-cleanup] {stage}", flush=True),
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
