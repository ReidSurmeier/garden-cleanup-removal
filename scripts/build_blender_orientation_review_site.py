from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.blender_orientation_review_site import (  # noqa: E402
    build_orientation_review_site,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: build_blender_orientation_review_site.py BATCH_ROOT"
        )
    batch_root = Path(sys.argv[1]).resolve()
    result = build_orientation_review_site(
        batch_root,
        batch_root / "viewer.html",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
