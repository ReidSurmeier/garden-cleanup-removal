from __future__ import annotations

import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import Metashape

from railing_removal.metashape_inventory import build_camera_inventory


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: metashape_camera_inventory.py PROJECT.psx OUTPUT.json"
        )
    project = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite inventory: {output}")
    document = Metashape.Document()
    document.open(str(project), read_only=True)
    if not bool(document.read_only):
        raise RuntimeError("Metashape did not open the project read-only")
    chunk = document.chunks[0]
    report = build_camera_inventory(
        chunk,
        project=project,
        metashape_version=str(Metashape.version),
        project_opened_read_only=bool(document.read_only),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(json.dumps({key: report[key] for key in (
        "metashape_version",
        "project_opened_read_only",
        "point_count",
        "camera_count",
        "aligned_camera_count",
        "depth_map_count",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
