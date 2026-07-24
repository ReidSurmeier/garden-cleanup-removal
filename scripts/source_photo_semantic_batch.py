from __future__ import annotations

import argparse
import json
from pathlib import Path

from plant_cleanup.dense_semantic import HuggingFaceOneFormerPredictor
from plant_cleanup.source_photo_batch import (
    run_source_photo_semantic_batch,
)


MODEL = "shi-labs/oneformer_ade20k_swin_large"
PLANT_LABELS = (
    "plant",
    "tree",
    "grass",
    "flower",
    "palm, palm tree",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Back-project real-photo plant semantics over a scan batch."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("canonical_root", type=Path)
    parser.add_argument("native_root", type=Path)
    parser.add_argument("inventory_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--camera-count", type=int, default=12)
    parser.add_argument("--maximum-dimension", type=int, default=768)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--torch-threads", type=int)
    args = parser.parse_args()
    if args.torch_threads is not None:
        import torch

        torch.set_num_threads(args.torch_threads)
    predictor = HuggingFaceOneFormerPredictor(
        MODEL,
        device=args.device,
    )
    report = run_source_photo_semantic_batch(
        args.manifest,
        args.canonical_root,
        args.native_root,
        args.inventory_root,
        args.output_root,
        predictor=predictor,
        model_id=MODEL,
        plant_labels=PLANT_LABELS,
        stride=args.stride,
        camera_count=args.camera_count,
        maximum_dimension=args.maximum_dimension,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
