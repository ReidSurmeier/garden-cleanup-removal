from __future__ import annotations

import argparse
import json
from pathlib import Path

from plant_cleanup.adaptive_profile import build_adaptive_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a source-bound adaptive cleanup config."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite config: {args.output.resolve()}"
        )
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    config = build_adaptive_config(args.source, base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
