from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.quality_report import build_quality_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-triage", type=Path)
    args = parser.parse_args()
    report = build_quality_report(
        args.batch_report,
        args.output,
        reference_triage_path=args.reference_triage,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
