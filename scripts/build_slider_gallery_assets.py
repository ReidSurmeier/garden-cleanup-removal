from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.slider_gallery import (  # noqa: E402
    build_slider_gallery_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_report", type=Path)
    parser.add_argument("publication_plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args()
    result = build_slider_gallery_assets(
        args.batch_report,
        args.publication_plan,
        args.output,
        max_width=args.max_width,
        quality=args.quality,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
