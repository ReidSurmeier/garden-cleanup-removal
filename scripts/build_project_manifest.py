from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.project_inventory import (  # noqa: E402
    build_project_manifest,
)


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "usage: build_project_manifest.py "
            "SOURCE_ROOT OUTPUT.json [PHOTO_ROOT]"
        )
    manifest = build_project_manifest(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        photo_root=Path(sys.argv[3]) if len(sys.argv) == 4 else None,
    )
    print(
        f"indexed {manifest['project_count']} projects beneath "
        f"{manifest['source_root']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
