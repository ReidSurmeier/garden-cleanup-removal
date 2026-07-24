from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.batch_overlay import build_batch_overlay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay reviewed correction reports onto a complete base batch."
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Base report followed by correction reports in precedence order.",
    )
    args = parser.parse_args()
    overlay = build_batch_overlay(args.reports, args.output)
    print(json.dumps(overlay["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
