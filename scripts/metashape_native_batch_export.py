from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import Metashape  # type: ignore[import-not-found]  # noqa: E402

from railing_removal.native_batch import run_native_export_batch  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: metashape_native_batch_export.py "
            "PROJECTS.json OUTPUT_ROOT"
        )
    report = run_native_export_batch(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Metashape,
        progress=lambda message: print(message, flush=True),
    )
    print(f"native batch summary: {report['summary']}", flush=True)


if __name__ == "__main__":
    main()
