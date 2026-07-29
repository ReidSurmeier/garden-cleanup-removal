from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.blender_orientation_production import (  # noqa: E402
    build_production_orientation_manifest,
)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: build_blender_production_manifest.py "
            "PROJECTS.json PRODUCTION_ROOT OUTPUT.json"
        )
    manifest = build_production_orientation_manifest(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
    )
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
