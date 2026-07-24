from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.photo_review import build_reference_review  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: build_reference_review.py PROJECTS.json OUTPUT_ROOT"
        )
    report = build_reference_review(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"reference review summary: {report['summary']}", flush=True)


if __name__ == "__main__":
    main()
