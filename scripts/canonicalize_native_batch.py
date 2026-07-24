from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.native_batch import (  # noqa: E402
    run_canonicalize_native_batch,
)


def main() -> None:
    if len(sys.argv) not in {4, 5}:
        raise SystemExit(
            "usage: canonicalize_native_batch.py "
            "PROJECTS.json NATIVE_ROOT OUTPUT_ROOT [STRIDE]"
        )
    report = run_canonicalize_native_batch(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        stride=int(sys.argv[4]) if len(sys.argv) == 5 else 1,
        progress=lambda message: print(message, flush=True),
    )
    print(f"canonical batch summary: {report['summary']}", flush=True)


if __name__ == "__main__":
    main()
