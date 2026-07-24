from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.adaptive_batch import build_adaptive_batch  # noqa: E402


def main() -> None:
    if len(sys.argv) not in {5, 6}:
        raise SystemExit(
            "usage: build_adaptive_batch.py PROJECTS.json CANONICAL_ROOT "
            "OUTPUT_ROOT BASE_CONFIG.json [STRIDE]"
        )
    report = build_adaptive_batch(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        Path(sys.argv[4]),
        stride=int(sys.argv[5]) if len(sys.argv) == 6 else 8,
        progress=lambda message: print(message, flush=True),
    )
    print(f"adaptive batch summary: {report['summary']}", flush=True)


if __name__ == "__main__":
    main()
