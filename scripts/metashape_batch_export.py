from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import Metashape  # type: ignore[import-not-found]  # noqa: E402

from railing_removal.metashape_batch_export import (  # noqa: E402
    run_metashape_export_batch,
)


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "usage: metashape_batch_export.py "
            "PROJECTS.json OUTPUT_ROOT [STRIDE]"
        )
    report = run_metashape_export_batch(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Metashape,
        stride=int(sys.argv[3]) if len(sys.argv) == 4 else 1,
        progress=lambda message: print(message, flush=True),
    )
    print(
        "batch export summary:",
        report["summary"],
        flush=True,
    )


if __name__ == "__main__":
    main()
