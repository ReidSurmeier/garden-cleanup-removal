from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.publish_outputs import publish_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_manifest", type=Path)
    parser.add_argument("batch_report", type=Path)
    parser.add_argument("publication_plan", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = publish_outputs(
        args.project_manifest,
        args.batch_report,
        args.publication_plan,
        args.report,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
