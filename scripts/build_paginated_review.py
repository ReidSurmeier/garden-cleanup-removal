from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.batch_review import build_paginated_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--page-size", type=int, default=20)
    args = parser.parse_args()
    report = build_paginated_review(
        args.batch_report,
        args.output,
        page_size=args.page_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
