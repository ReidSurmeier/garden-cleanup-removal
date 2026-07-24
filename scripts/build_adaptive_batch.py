from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.adaptive_batch import build_adaptive_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("projects", type=Path)
    parser.add_argument("canonical_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("base_config", type=Path)
    parser.add_argument("stride", type=int, nargs="?", default=8)
    parser.add_argument("--scene-catalog", type=Path)
    args = parser.parse_args()
    report = build_adaptive_batch(
        args.projects,
        args.canonical_root,
        args.output_root,
        args.base_config,
        stride=args.stride,
        scene_catalog_path=args.scene_catalog,
        progress=lambda message: print(message, flush=True),
    )
    print(f"adaptive batch summary: {report['summary']}", flush=True)


if __name__ == "__main__":
    main()
