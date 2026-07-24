from __future__ import annotations

import argparse
import json
from pathlib import Path

from railing_removal.batch_review import build_batch_review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a non-overwriting visual review for a completed batch."
    )
    parser.add_argument("batch_report", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_batch_review(args.batch_report),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
