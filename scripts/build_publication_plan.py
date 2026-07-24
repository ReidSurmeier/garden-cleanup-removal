from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.publication_plan import (  # noqa: E402
    build_publication_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an exhaustive additive publication plan."
    )
    parser.add_argument("project_manifest", type=Path)
    parser.add_argument("batch_report", type=Path)
    parser.add_argument("reference_triage", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--artifact-tag", required=True)
    args = parser.parse_args()
    plan = build_publication_plan(
        args.project_manifest,
        args.batch_report,
        args.reference_triage,
        args.output,
        artifact_tag=args.artifact_tag,
    )
    print(json.dumps(plan["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
