from __future__ import annotations

import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import Metashape  # type: ignore[import-not-found]  # noqa: E402

from railing_removal.metashape_reader import (  # noqa: E402
    export_reader_cloud_readonly,
)


def main() -> None:
    if len(sys.argv) not in {4, 5}:
        raise SystemExit(
            "usage: metashape_reader_export.py "
            "PROJECT.psx OUTPUT.ply REPORT.json [STRIDE]"
        )
    report_path = Path(sys.argv[3])
    if report_path.exists():
        raise FileExistsError(
            f"refusing to overwrite report: {report_path.resolve()}"
        )
    report = export_reader_cloud_readonly(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Metashape,
        stride=int(sys.argv[4]) if len(sys.argv) == 5 else 1,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
