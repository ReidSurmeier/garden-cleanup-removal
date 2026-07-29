from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import Metashape

from railing_removal.camera_inventory_batch import (
    run_camera_inventory_batch,
)
from railing_removal.metashape_inventory import build_camera_inventory


def _inventory_project(project: Path) -> dict[str, Any]:
    document = Metashape.Document()
    try:
        document.open(str(project), read_only=True)
        if not bool(document.read_only):
            raise RuntimeError("Metashape did not open the project read-only")
        chunk = document.chunks[0]
        return build_camera_inventory(
            chunk,
            project=project,
            metashape_version=str(Metashape.version),
            project_opened_read_only=bool(document.read_only),
        )
    finally:
        del document
        gc.collect()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: metashape_camera_inventory_batch.py "
            "PROJECTS.json OUTPUT_ROOT"
        )
    report = run_camera_inventory_batch(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        inventory_project=_inventory_project,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
