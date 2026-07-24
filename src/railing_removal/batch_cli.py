from __future__ import annotations

import argparse
import json
from pathlib import Path

from railing_removal.batch import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a non-overwriting cleanup batch with cached models."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run_batch(
        args.manifest,
        args.output,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
